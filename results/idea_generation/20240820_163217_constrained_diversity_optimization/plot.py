import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import pandas as pd
import scipy.stats

# LOAD FINAL RESULTS:
folders = os.listdir("./")
final_results = {}
all_results_info = {}

for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        all_results = pd.read_csv(osp.join(folder, "all_results.csv"))
        all_results_info[folder] = all_results

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
}

def get_best_run(all_results_info):
    run_numbers = [int(run.split('_')[1]) for run in all_results_info.keys() if run != 'run_0']
    return f"run_{max(run_numbers)}" if run_numbers else None

def create_comparison_plots(all_results_info, labels):
    metrics = ["Overall", "Soundness", "Presentation", "Contribution", "Originality", "Quality", "Clarity", "Significance"]
    
    best_run = get_best_run(all_results_info)
    if best_run is None:
        print("No runs other than baseline found.")
        return
    
    runs_to_compare = ['run_0', best_run]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))  # Increased figure height
    fig.suptitle("Comparison of Review Results: Baseline vs Best Run", fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 4, i % 4]
        
        data = [all_results_info[run][metric].values for run in runs_to_compare]
        
        bp = ax.boxplot(data, labels=[labels.get(run, run) for run in runs_to_compare])
        
        for j, box in enumerate(bp['boxes']):
            box.set(color='blue' if j == 0 else 'red', linewidth=2)
        
        ax.set_title(metric, fontsize=10)
        ax.set_ylim(0, 10)
        
        # Rotate x-axis labels and adjust their position
        ax.set_xticklabels([labels.get(run, run) for run in runs_to_compare], rotation=45, ha='right')
        
        # Adjust y-axis label font size
        ax.tick_params(axis='y', labelsize=8)
        
        baseline = all_results_info['run_0'][metric].values
        current = all_results_info[best_run][metric].values
        _, p_value = scipy.stats.ttest_ind(baseline, current)
        if p_value < 0.05:
            ax.text(2, ax.get_ylim()[1], '*', horizontalalignment='center', fontsize=12)
    
    plt.tight_layout()
    # Adjust the space between plots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig('comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_decision_comparison(all_results_info, labels):
    best_run = get_best_run(all_results_info)
    if best_run is None:
        print("No runs other than baseline found.")
        return
    
    runs_to_compare = ['run_0', best_run]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    decisions = []
    run_names = []
    
    for run in runs_to_compare:
        decision_counts = all_results_info[run]['Decision'].value_counts(normalize=True)
        decisions.append(decision_counts['Accept'] if 'Accept' in decision_counts else 0)
        run_names.append(labels.get(run, run))
    
    colors = ['blue', 'red']
    
    bars = ax.bar(run_names, decisions, color=colors)
    
    ax.set_ylabel('Acceptance Rate', fontsize=12)
    ax.set_title('Comparison of Acceptance Rates: Baseline vs Best Run', fontsize=14)
    ax.set_ylim(0, 1)
    
    # Rotate x-axis labels and adjust their position
    ax.set_xticklabels(run_names, rotation=45, ha='right')
    
    # Adjust font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('decision_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

create_comparison_plots(all_results_info, labels)
create_decision_comparison(all_results_info, labels)