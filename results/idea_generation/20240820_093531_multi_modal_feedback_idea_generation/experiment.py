import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pathlib
import openai
import pandas as pd
import numpy as np
import shutil
import json
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from datetime import datetime
# from idea_generation_module import generate_ideas, check_idea_novelty, select_best_idea  # TODO: experiment.py に移動する

import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append("/root/AI-Scientist") # TODO: base path を外から指定できるようにする
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import perform_writeup, generate_latex
from ai_scientist.perform_review import perform_review, load_paper

# 
import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append("/root/AI-Scientist")
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers


##################### METHOD #############################
import requests
import backoff

S2_API_KEY = os.getenv("S2_API_KEY")

idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

<idea_generation_module.py>
{idea_generation_code}
</idea_generation_module.py>

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
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).
- "Significance": A rating from 1 to 10 (lowest to highest).

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


# GENERATE IDEAS
def generate_ideas(
    base_dir,
    client,
    model,
    skip_generation=False,
    max_num_generations=20,
    num_reflections=5,
):
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
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
            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            # Iteratively improve task.
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
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert (
                        json_output is not None
                    ), "Failed to extract JSON from LLM output"
                    print(json_output)

                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break

            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


# GENERATE IDEAS OPEN-ENDED
def generate_next_idea(
    base_dir,
    client,
    model,
    prev_idea_archive=[],
    num_reflections=5,
    max_attempts=10,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        # seed the archive on the first run with pre-existing ideas
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # Iteratively improve task.
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
                        ## PARSE OUTPUT
                        json_output = extract_json_between_markers(text)
                        assert (
                            json_output is not None
                        ), "Failed to extract JSON from LLM output"
                        print(json_output)

                        if "I am done" in text:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

    ## SAVE IDEAS
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive


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
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers


novelty_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''


def check_idea_novelty(
    ideas,
    base_dir,
    client,
    model,
    max_num_iterations=2,
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                ## SEARCH FOR PAPERS
                query = json_output["Query"]
                papers = search_for_papers(query, result_limit=10)
                if papers is None:
                    papers_str = "No papers found."

                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        )
                    )
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["novel"] = novel

    # Save results to JSON file
    results_file = osp.join(base_dir, "ideas.json")
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas

def select_best_idea(ideas):
    max_score = float('-inf')
    best_idea = None
    
    for idea in reversed(ideas):
        total_score = (
            idea['Interestingness'] +
            idea['Novelty'] +
            idea['Significance']
        )
        
        if total_score > max_score:
            max_score = total_score
            best_idea = idea
    
    return best_idea
##################### METHOD #############################

##################### EXPERIMENT #############################
# Hardcoded parameters
EXPERIMENT = "2d_diffusion"
MODEL = "gpt-4o-2024-05-13"
# MODEL = "claude-3-5-sonnet-20240620"
NUM_IDEAS = 10
NUM_REFLECTIONS = 3

def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def worker(queue, base_dir, results_dir, client, client_model):
    print(f"Worker started.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(base_dir, results_dir, idea, client, client_model)
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    print(f"Worker finished.")

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

if MODEL == "claude-3-5-sonnet-20240620":
    import anthropic
    client = anthropic.Anthropic()
elif MODEL == "gpt-4o-2024-05-13":
    client = openai.OpenAI()
else:
    raise ValueError(f"Model {MODEL} not supported.")
client_model = MODEL

# base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "templates", EXPERIMENT)
# results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", EXPERIMENT)
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
    skip_generation=False,
    max_num_generations=NUM_IDEAS,
    num_reflections=NUM_REFLECTIONS,
)
# ideas = check_idea_novelty(
#     ideas,
#     base_dir=base_dir,
#     client=client,
#     model=client_model,
# )
# TODO: semantic scholar takes too much time

with open(os.path.join(base_dir, "ideas.json"), "w") as f:
    json.dump(ideas, f, indent=4)

# novel_ideas = [idea for idea in ideas if idea["novel"]]
# novel_ideas = ideas  # TODO: semantic scholar takes too much time

idea = select_best_idea(ideas)

# for idea in novel_ideas:
print(f"Processing idea: {idea['Name']}")
success = do_idea(base_dir, results_dir, idea, client, client_model)
print(f"Completed idea: {idea['Name']}, Success: {success}")

print("All ideas evaluated.")


# Save results
import argparse
parser = argparse.ArgumentParser()  
parser.add_argument("--out_dir", type=str, default="run_0")
args = parser.parse_args()

out_dir = args.out_dir
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

LLM_COLS = [
    "paper_id",
    "Summary",
    "Questions",
    "Limitations",
    "Ethical Concerns",
    "Soundness",
    "Presentation",
    "Contribution",
    "Overall",
    "Confidence",
    "Strengths",
    "Weaknesses",
    "Originality",
    "Quality",
    "Clarity",
    "Significance",
    "Decision",
]

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
                    
                    # Create a dictionary to store the values for this paper
                    paper_data = {}
                    
                    # Iterate through the columns and add data if it exists in the review
                    for col in LLM_COLS:
                        if col in review:
                            paper_data[col] = review[col]
                    
                    # Add the data to the DataFrame
                    llm_ratings.loc[paper_name] = paper_data
                    
                    print(f"Review for {paper_name} saved.")
            except json.JSONDecodeError:
                print(f"Error reading review file for {paper_name}. Skipping.")
        else:
            print(f"No review.txt found in {subdir}. Skipping this idea.")

def process_decision(x):
    return 1 if x == "Accept" else 0 if x == "Reject" else np.nan

def process_ethical_concerns(x):
    return 1 if x else 0 if x is False else np.nan

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
##################### EXPERIMENT #############################