import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import pandas as pd
import shutil
import json
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from datetime import datetime
from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import perform_writeup, generate_latex
from ai_scientist.perform_review import perform_review, load_paper

# Hardcoded parameters
EXPERIMENT = "nanoGPT"
MODEL = "gpt-4o-2024-05-13"
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

client_model = "gpt-4o-2024-05-13"
client = openai.OpenAI()
client_model = MODEL

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", EXPERIMENT)
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", EXPERIMENT)
os.makedirs(results_dir, exist_ok=True)

ideas = generate_ideas(
    base_dir,
    client=client,
    model=client_model,
    skip_generation=False,
    max_num_generations=NUM_IDEAS,
    num_reflections=NUM_REFLECTIONS,
)
ideas = check_idea_novelty(
    ideas,
    base_dir=base_dir,
    client=client,
    model=client_model,
)

with open(os.path.join(base_dir, "ideas.json"), "w") as f:
    json.dump(ideas, f, indent=4)

novel_ideas = [idea for idea in ideas if idea["novel"]]

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
            with open(review_file_path, "r") as f:
                review = json.load(f)
                llm_ratings.loc[paper_name] = review
        else:
            print(f"No review.txt found in {subdir}")

final_info = {
    "means": str(llm_ratings.mean())
}

with open(os.path.join(out_dir, "final_info.json"), "w") as f:
    json.dump(final_info, f)

llm_ratings.to_csv(os.path.join(out_dir, "all_results.csv"))