import matplotlib.pyplot as plt
import scipy.stats
import json
import os
import os.path as osp
import pandas as pd

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
    "run_1": "Ethical Prompt",
    "run_2": "Ethical Checklist",
    "run_3": "Reduced Ideas",
    "run_4": "Detailed Guidelines",
    "run_5": "Additional Reflections",
}

# LOAD FINAL RESULTS:
folders = os.listdir("./")
final_results = {}
all_results_info = {}

for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        all_results = pd.read_csv(osp.join(folder, "all_results.csv"))
        if folder in labels:
            all_results_info[folder] = all_results


def create_comparison_plots(all_results_info, labels):
    # Metrics to plot
    metrics = ["Overall", "Soundness", "Presentation", "Contribution", "Originality", "Quality", "Clarity", "Significance"]
    
    # Set up the plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Comparison of Review Results across Different Runs", fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 4, i % 4]
        
        data = []
        for run, results in all_results_info.items():
            data.append(results[metric].values)
        
        # Create box plot
        bp = ax.boxplot(data, labels=[labels.get(run, run) for run in all_results_info.keys()])
        
        # Color the boxes
        for j, box in enumerate(bp['boxes']):
            if j == 0:  # Baseline (run_0)
                box.set(color='blue', linewidth=2)
            else:
                box.set(color='red', linewidth=2)
        
        ax.set_title(metric)
        ax.set_ylim(0, 10)  # Assuming scores are from 0 to 10
        
        # Add statistical significance
        baseline = all_results_info['run_0'][metric].values
        for j, run in enumerate(all_results_info.keys()):
            if run != 'run_0':
                current = all_results_info[run][metric].values
                _, p_value = scipy.stats.ttest_ind(baseline, current)
                if p_value < 0.05:
                    ax.text(j+1, ax.get_ylim()[1], '*', horizontalalignment='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

create_comparison_plots(all_results_info, labels)

# Decision Comparison
def create_decision_comparison(all_results_info, labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    decisions = []
    run_names = []
    
    for run, results in all_results_info.items():
        decision_counts = results['Decision'].value_counts(normalize=True)
        decisions.append(decision_counts['Accept'] if 'Accept' in decision_counts else 0)
        run_names.append(labels.get(run, run))
    
    colors = ['blue' if run == 'run_0' else 'red' for run in all_results_info.keys()]
    
    bars = ax.bar(run_names, decisions, color=colors)
    
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Comparison of Acceptance Rates across Different Runs')
    ax.set_ylim(0, 1)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('decision_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

create_decision_comparison(all_results_info, labels)
