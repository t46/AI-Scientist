import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Union

import openai
import anthropic
import pandas as pd
import numpy as np
import requests
import backoff
import argparse
import pathlib
import shutil
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence_transformers module not found. Using a simple fallback for embeddings.")
    
    class SimpleSentenceTransformer:
        def encode(self, sentences):
            return np.array([hash(s) for s in sentences]).reshape(-1, 1)
    
    SentenceTransformer = SimpleSentenceTransformer

sys.path.append("/root/AI-Scientist")
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_review import perform_review, load_paper
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

# Constants
EXPERIMENT = "2d_diffusion"
# MODEL = "gpt-4o-2024-05-13"
MODEL = "claude-3-5-sonnet-20240620"
NUM_IDEAS = 5
NUM_REFLECTIONS = 2
NUM_CRITIQUE_REFINEMENT_CYCLES = 3  # New constant for ARSCRS
LLM_COLS = ["paper_id", "Summary", "Questions", "Limitations", "Ethical Concerns", "Soundness", "Presentation", "Contribution", "Overall", "Confidence", "Strengths", "Weaknesses", "Originality", "Quality", "Clarity", "Significance", "Decision"]


# Prompts
idea_first_prompt = """
{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for method, research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.
You should think about the problems and solutions that you can find in the current idea generation process, and how to improve them by proposing new ideas.
Note that this is not mere task solving but academic research, so you should focus on the novelty, significance, feasibility, interestingness, and critical evaluation of the ideas.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Method": A description of the new method proposed in this research.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest). Whether it is intellectually stimulating and piques scientists' curiosity.
- "Feasibility": A rating from 1 to 10 (lowest to highest). The feasibility of this idea considering that AI would execute it completely autonomously, without any human intervention, interaction with the internet, or preparation of new data.
- "Novelty": A rating from 1 to 10 (lowest to highest). Not only did it simply not exist, but something so innovative that no one, not even researchers seeking new discoveries, has ever seen before is considered to be novel.
- "Significance": A rating from 1 to 10 (lowest to highest). Whether it has a significant impact on the research field or the practical impact on the world.

Be cautious, critical, harsh and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, significance, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""

def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )

@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": os.getenv("S2_API_KEY")},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(f"Response Content: {rsp.text[:500]}")
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers

def generate_ideas(base_dir, client, model, max_num_generations=20, num_reflections=5):
    idea_str_archive = []
    with open(os.path.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(os.path.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(os.path.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    ideas = []
    for _ in range(max_num_generations):
        print(f"\nGenerating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    print(json_output)

                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break

            ideas.append(json_output)
            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    with open(os.path.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas

def critique_ideas(ideas, client, model):
    critique_prompt = """
    Analyze the following research idea for potential weaknesses, contradictions, or areas for improvement. Consider both internal logic and external research criteria. Provide a detailed critique focusing on:

    1. Methodology: Is the proposed method sound and well-justified?
    2. Feasibility: Can this idea be realistically implemented given the constraints?
    3. Novelty: How original is this idea compared to existing research?
    4. Significance: What is the potential impact of this research?
    5. Limitations: What are the potential drawbacks or challenges?

    Research Idea:
    {idea}

    Provide your critique in the following JSON format:
    ```json
    {{
        "methodology_critique": "string",
        "feasibility_critique": "string",
        "novelty_critique": "string",
        "significance_critique": "string",
        "limitations": "string",
        "overall_assessment": "string",
        "improvement_suggestions": ["string", "string", ...]
    }}
    ```
    """

    critiques = []
    for idea in ideas:
        try:
            response, _ = get_response_from_llm(
                critique_prompt.format(idea=json.dumps(idea, indent=2)),
                client=client,
                model=model,
                system_message="You are a critical research evaluator. Provide honest and constructive feedback.",
            )
            critique = json.loads(response)
            critiques.append(critique)
        except Exception as e:
            print(f"Failed to critique idea: {e}")
            critiques.append(None)

    return critiques

def refine_ideas(ideas, critiques, client, model):
    refine_prompt = """
    Given the original research idea and its critique, refine the idea to address the identified weaknesses and improve its overall quality. Focus on enhancing the methodology, feasibility, novelty, and significance while addressing the limitations.

    Original Idea:
    {original_idea}

    Critique:
    {critique}

    Provide the refined idea in the same JSON format as the original, but with improvements based on the critique:
    ```json
    {{
        "Name": "string",
        "Title": "string",
        "Method": "string",
        "Experiment": "string",
        "Interestingness": int,
        "Feasibility": int,
        "Novelty": int,
        "Significance": int
    }}
    ```

    Ensure that the refined idea addresses the critique while maintaining or improving upon the strengths of the original idea.
    """

    refined_ideas = []
    for idea, critique in zip(ideas, critiques):
        if critique is None:
            refined_ideas.append(idea)
            continue

        try:
            response, _ = get_response_from_llm(
                refine_prompt.format(
                    original_idea=json.dumps(idea, indent=2),
                    critique=json.dumps(critique, indent=2)
                ),
                client=client,
                model=model,
                system_message="You are an expert research idea refiner. Improve ideas based on critiques while maintaining their core essence.",
            )
            refined_idea = json.loads(response)
            refined_ideas.append(refined_idea)
        except Exception as e:
            print(f"Failed to refine idea: {e}")
            refined_ideas.append(idea)

    return refined_ideas

def calculate_diversity(ideas, model):
    embeddings = model.encode([idea['Method'] + ' ' + idea['Experiment'] for idea in ideas])
    similarity_matrix = cosine_similarity(embeddings)
    diversity_scores = 1 - similarity_matrix.mean(axis=1)
    return diversity_scores

def select_diverse_ideas(ideas, num_to_select):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except:
        print("Warning: Using SimpleSentenceTransformer fallback.")
        model = SentenceTransformer()
    
    diversity_scores = calculate_diversity(ideas, model)
    
    combined_scores = np.array([
        sum(idea[key] for key in ['Interestingness', 'Feasibility', 'Novelty', 'Significance']) + diversity_score
        for idea, diversity_score in zip(ideas, diversity_scores)
    ])
    
    selected_indices = np.argsort(combined_scores)[-num_to_select:]
    return [ideas[i] for i in selected_indices]

def evaluate_idea_improvement(original_idea, refined_idea):
    score_keys = ['Interestingness', 'Feasibility', 'Novelty', 'Significance']
    original_score = sum(original_idea[key] for key in score_keys)
    refined_score = sum(refined_idea[key] for key in score_keys)
    return (refined_score - original_score) / original_score  # Return improvement percentage

def do_idea(base_dir, results_dir, idea, client, client_model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = os.path.join(results_dir, idea_name)
    assert not os.path.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    
    with open(os.path.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    
    exp_file = os.path.join(folder_name, "experiment.py")
    vis_file = os.path.join(folder_name, "plot.py")
    notes = os.path.join(folder_name, "notes.txt")
    
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")

    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")

        fnames = [exp_file, vis_file, notes]
        io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
        main_model = Model(MODEL)
        coder = Coder.create(
            main_model=main_model, fnames=fnames, io=io, stream=False, use_git=False, edit_format="diff"
        )

        print_time()
        print(f"*Starting Experiments*")
        success = perform_experiments(idea, folder_name, coder, baseline_results)
        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False

        print_time()
        print(f"*Starting Writeup*")
        writeup_file = os.path.join(folder_name, "latex", "template.tex")
        fnames = [exp_file, writeup_file, notes]
        coder = Coder.create(
            main_model=main_model, fnames=fnames, io=io, stream=False, use_git=False, edit_format="diff"
        )
        perform_writeup(idea, folder_name, coder, client, client_model)

        print_time()
        print(f"*Starting Review*")
        paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
        review = perform_review(
            paper_text,
            model=MODEL,
            client=client,
            num_reflections=1,
            num_fs_examples=1,
            num_reviews_ensemble=5,
            temperature=0.1,
        )
        with open(os.path.join(folder_name, "review.txt"), "w") as f:
            f.write(json.dumps(review, indent=4))

        return True
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False

def process_decision(x):
    return 1 if x == "Accept" else 0 if x == "Reject" else np.nan

def process_ethical_concerns(x):
    return 1 if x else 0 if x is False else np.nan

def main():
    if MODEL == "claude-3-5-sonnet-20240620":
        client = anthropic.Anthropic()
    elif MODEL == "gpt-4o-2024-05-13":
        client = openai.OpenAI()
    else:
        raise ValueError(f"Model {MODEL} not supported.")
    client_model = MODEL

    base_dir = os.path.join("/root/AI-Scientist/templates", EXPERIMENT)
    results_dir = os.path.join("/root/AI-Scientist/results", EXPERIMENT)
    os.makedirs(results_dir, exist_ok=True)

    archived_results_dir = os.path.join("/root/AI-Scientist/archived_results", EXPERIMENT)
    os.makedirs(archived_results_dir, exist_ok=True)
    for subdir in os.listdir(results_dir):
        shutil.move(os.path.join(results_dir, subdir), archived_results_dir)

    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        max_num_generations=NUM_IDEAS,
        num_reflections=NUM_REFLECTIONS,
    )

    # ARSCRS: Adaptive Recursive Self-Critique and Refinement System
    for cycle in range(NUM_CRITIQUE_REFINEMENT_CYCLES):
        print(f"Starting ARSCRS cycle {cycle + 1}/{NUM_CRITIQUE_REFINEMENT_CYCLES}")
        
        critiques = critique_ideas(ideas, client, client_model)
        refined_ideas = refine_ideas(ideas, critiques, client, client_model)
        
        # Evaluate improvement
        improvements = [evaluate_idea_improvement(original, refined) 
                        for original, refined in zip(ideas, refined_ideas)]
        avg_improvement = sum(improvements) / len(improvements)
        print(f"Average improvement in cycle {cycle + 1}: {avg_improvement:.2%}")
        
        # Select diverse ideas
        ideas = select_diverse_ideas(refined_ideas, num_to_select=min(len(refined_ideas), NUM_IDEAS))
        
        print(f"Number of ideas after diversity selection: {len(ideas)}")
        
        with open(os.path.join(base_dir, f"ideas_cycle_{cycle + 1}.json"), "w") as f:
            json.dump(ideas, f, indent=4)

    # Select the best idea from the final set
    idea = max(ideas, key=lambda x: sum(x[key] for key in ['Interestingness', 'Feasibility', 'Novelty', 'Significance']))

    print(f"Processing best idea: {idea['Name']}")
    success = do_idea(base_dir, results_dir, idea, client, client_model)
    print(f"Completed idea: {idea['Name']}, Success: {success}")

    print("ARSCRS process completed.")

    # Save results
    parser = argparse.ArgumentParser()  
    parser.add_argument("--out_dir", type=str, default="run_0")
    args = parser.parse_args()

    out_dir = args.out_dir
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    llm_ratings = pd.DataFrame(columns=LLM_COLS)
    llm_ratings.set_index("paper_id", inplace=True)

    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        
        if os.path.isdir(subdir_path):
            review_file_path = os.path.join(subdir_path, 'review.txt')
            paper_name = subdir
            
            if os.path.isfile(review_file_path):
                try:
                    with open(review_file_path, "r") as f:
                        review = json.load(f)
                        
                        paper_data = {col: review.get(col) for col in LLM_COLS if col in review}
                        
                        llm_ratings.loc[paper_name] = paper_data
                        print(f"Review for {paper_name} saved.")

                except json.JSONDecodeError:
                    print(f"Error reading review file for {paper_name}. Skipping.")
            else:
                print(f"No review.txt found in {subdir}. Skipping this idea.")

    if not llm_ratings.empty:
        llm_ratings.to_csv(os.path.join(out_dir, "all_results.csv"))
        print(f"Results saved to {out_dir}")

        numeric_cols = llm_ratings.select_dtypes(include=[np.number]).columns.tolist()

        if "Decision" in llm_ratings.columns:
            llm_ratings["Decision_Numeric"] = llm_ratings["Decision"].apply(process_decision)
            numeric_cols.append("Decision_Numeric")

        if "Ethical Concerns" in llm_ratings.columns:
            llm_ratings["Ethical_Concerns_Numeric"] = llm_ratings["Ethical Concerns"].apply(process_ethical_concerns)
            numeric_cols.append("Ethical_Concerns_Numeric")

        final_info = {
            "result": {
                "means": llm_ratings[numeric_cols].mean().to_dict()
            }
        }

        with open(os.path.join(out_dir, "final_info.json"), "w") as f:
            json.dump(final_info, f, indent=4)
    else:
        print("No valid results to save. Check if any reviews were processed successfully.")

if __name__ == "__main__":
    main()
