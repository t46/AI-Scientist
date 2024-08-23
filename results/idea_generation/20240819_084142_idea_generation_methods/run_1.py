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
from idea_generation_module import generate_ideas, check_idea_novelty

import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append("/root/AI-Scientist") # TODO: base path を外から指定できるようにする
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import perform_writeup, generate_latex
from ai_scientist.perform_review import perform_review, load_paper

# Hardcoded parameters
EXPERIMENT = "2d_diffusion"
MODEL = "gpt-4o-2024-05-13"
# MODEL = "claude-3-5-sonnet-20240620"
NUM_IDEAS = 2
NUM_REFLECTIONS = 1

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

# import anthropic
# client_model = "claude-3-5-sonnet-20240620"
# client = anthropic.Anthropic()
client = openai.OpenAI()
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
    num_reflections=NUM_REFLECTIONS
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
novel_ideas = ideas  # TODO: semantic scholar takes too much time

for idea in novel_ideas:
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
