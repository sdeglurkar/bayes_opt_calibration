import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pick_to_learn_settings import *

# -----------------------------
# Setup
# -----------------------------
conditions = ["Boundary, Slice 1", "Boundary, Slice 2", \
                "Random, Slice 1", "Random, Slice 2"]
baselines = ["LB Iterative",r"LB Robust, Min $\epsilon$",r"LB Robust, Min $N$",\
            r"LB Robust, Min $\alpha$",r"LB Robust, Median $\epsilon$"]
methods = ["Our Method"]
all_methods = methods + baselines
num_methods = len(all_methods)
num_conditions = len(conditions)

# Example data for 3 dimensions
samples_2d = np.array([
    [191.1, 290.4, 1000.0, 150.0, 415.0, 400.0],
    [193.625, 462.0, 393.75, 150.0, 412.5, 537.5],
    [190.4, 290.4, 1000.0, 150.0, 415.0, 400.0],
    [190.75, 396.0, 475.0, 150.0, 550.0, 362.5],
])

samples_3d = np.array([
    [350, 400, 300, 380, 360, 370],
    [370, 420, 310, 390, 370, 385],
    [360, 410, 320, 400, 365, 375],
    [380, 430, 330, 410, 375, 395],
])

samples_4d = np.array([
    [350, 400, 300, 380, 360, 370],
    [370, 420, 310, 390, 370, 385],
    [360, 410, 320, 400, 365, 375],
    [380, 430, 330, 410, 375, 395],
])

fpr_2d = np.array([
    [0.0058823529411764705, 0.017993079584775088, 0.0038062283737024223, 0.010899653979238755, 0.08823529411764706, 0.03944636678200692],
    [0.02784653465346535, 0.005775577557755776, 0.020214521452145216, 0.026815181518151814, 0.2838283828382838, 0.24628712871287128],
    [0.014705882352941175, 0.017993079584775088, 0.0038062283737024223, 0.010899653979238755, 0.08823529411764706, 0.03944636678200692],
    [0.0627062706270627, 0.0066006600660066, 0.018151815181518153, 0.025165016501650164, 0.2838283828382838, 0.25165016501650167],
])

fpr_3d = np.array([
    [0.05, 0.07, 0.06, 0.04, 0.05, 0.03],
    [0.06, 0.08, 0.05, 0.05, 0.06, 0.04],
    [0.05, 0.06, 0.07, 0.06, 0.05, 0.03],
    [0.04, 0.07, 0.06, 0.05, 0.04, 0.02],
])

fpr_4d = np.array([
    [0.05, 0.07, 0.06, 0.04, 0.05, 0.03],
    [0.06, 0.08, 0.05, 0.05, 0.06, 0.04],
    [0.05, 0.06, 0.07, 0.06, 0.05, 0.03],
    [0.04, 0.07, 0.06, 0.05, 0.04, 0.02],
])

fnr_2d = np.array([
    [0.016782334384858044, 0.06971608832807571, 0.1583596214511041, 0.09911671924290222, 0.0031545741324921135, 0.019558359621451103],
    [0.04447655748233783, 0.6500481695568401, 0.40815671162491973, 0.35589274245343616, 0.010918432883750802, 0.023603082851637765],
    [0.023280757097791795, 0.06971608832807571, 0.1583596214511041, 0.09911671924290222, 0.0031545741324921135, 0.019558359621451103],
    [0.051541425818882464, 0.6454720616570327, 0.426461143224149, 0.3683365446371227, 0.010918432883750802, 0.021355170199100833],
])

fnr_3d = np.array([
    [0.07, 0.06, 0.08, 0.05, 0.07, 0.03],
    [0.08, 0.07, 0.09, 0.06, 0.08, 0.04],
    [0.06, 0.08, 0.07, 0.06, 0.06, 0.03],
    [0.07, 0.05, 0.08, 0.05, 0.07, 0.02],
])

fnr_4d = np.array([
    [0.07, 0.06, 0.08, 0.05, 0.07, 0.03],
    [0.08, 0.07, 0.09, 0.06, 0.08, 0.04],
    [0.06, 0.08, 0.07, 0.06, 0.06, 0.03],
    [0.07, 0.05, 0.08, 0.05, 0.07, 0.02],
])

# Success counts
successes = np.array([
    [10, np.nan, np.nan, np.nan, np.nan, np.nan],
    [8, np.nan, np.nan, np.nan, np.nan, np.nan],
    [10, np.nan, np.nan, np.nan, np.nan, np.nan],
    [4, np.nan, np.nan, np.nan, np.nan, np.nan],
])

# -----------------------------
# Function to plot Number of Samples
# -----------------------------
def plot_samples(samples, dim_name):
    x = np.arange(num_conditions)
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(8,5))
    for i, method in enumerate(all_methods):
        bars = ax.bar(
            x + (i - num_methods/2)*width + width/2,
            samples[:, i],
            width,
            label=method,
            color=f"C{i}" if method != "Actual" else "red",
            alpha=0.8 if method != "Actual" else 1.0,
            hatch='//' if method=="Actual" else ''
        )
        labels = []
        for c in range(num_conditions):
            if not np.isnan(successes[c,i]):
                labels.append(f"{successes[c,i]}/10")
            else:
                labels.append("")
        # labels = [f"{successes[c,i]}/10" for c in range(num_conditions)]
        ax.bar_label(bars, labels=labels, padding=2, fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30)
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Number of Samples ({dim_name})")
    # ax.legend(loc='upper center', ncol=num_methods, bbox_to_anchor=(0.5, 1.15))
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Function to plot FPR/FNR combined
# -----------------------------
def plot_fpr_fnr(fpr, fnr, dim_name):
    x = np.arange(num_conditions)
    width = 0.07
    
    fig, ax = plt.subplots(figsize=(8,5))
    colors = [f"C{i}" for i in range(num_methods)]
    
    for i, method in enumerate(all_methods):
        bars_fpr = ax.bar(
            # x + (i - num_methods/2)*width*2,
            x + (i - num_methods/2)*width*2,
            fpr[:, i],
            width,
            color=colors[i],
            alpha=0.7
        )
        bars_fnr = ax.bar(
            # x + (i - num_methods/2)*width*2 + width,
            x + (i - num_methods/2)*width*2 + width,
            fnr[:, i],
            width,
            color=colors[i],
            alpha=0.9,
            hatch='//'
        )
        # Add success labels
        labels = []
        for c in range(num_conditions):
            if not np.isnan(successes[c,i]):
                labels.append(f"{successes[c,i]}/10")
            else:
                labels.append("")
        # labels = [f"{successes[c,i]}/10" for c in range(num_conditions)]
        ax.bar_label(bars_fpr, labels=labels, padding=2, fontsize=8)
        # ax.bar_label(bars_fnr, labels=labels, padding=2, fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30)
    ax.set_ylabel("Rate")
    ax.set_title(f"FPR and FNR ({dim_name})")
    
    # Custom legend: FPR/FNR per method
    legend_elements = []
    for i, method in enumerate(all_methods):
        legend_elements.append(Patch(facecolor=colors[i], alpha=0.7, label=f"{method} FPR"))
        legend_elements.append(Patch(facecolor=colors[i], alpha=0.9, hatch='//', label=f"{method} FNR"))
    ax.legend(handles=legend_elements, loc='upper right', ncol=2)
    
    plt.tight_layout()
    plt.show()

# -----------------------------
# Plot for all dimensions
# -----------------------------
dimensions = [("2D", samples_2d, fpr_2d, fnr_2d),
              ("3D", samples_3d, fpr_3d, fnr_3d),
              ("4D", samples_4d, fpr_4d, fnr_4d)]

for dim_name, samples, fpr, fnr in dimensions:
    plot_samples(samples, dim_name)
    plot_fpr_fnr(fpr, fnr, dim_name)