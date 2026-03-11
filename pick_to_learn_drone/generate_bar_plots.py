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

# alpha = 0.05, tol_alpha = 0.03
samples_2d = np.array([
    [191.1, 290.4, 1000.0, 150.0, 415.0, 400.0],
    [193.625, 462.0, 393.75, 150.0, 412.5, 537.5],
    [190.4, 290.4, 1000.0, 150.0, 415.0, 400.0],
    [190.75, 396.0, 475.0, 150.0, 550.0, 362.5],
])

# alpha = 0.1, tol_alpha = 0.05
samples_3d = np.array([
    [172.4, 369.6, 1000.0, 150.0, 450.0, 415.0],
    [180.0, 440.0, 383.3333333333333, 150.0, 316.6666666666667, 383.3333333333333],
    [176.0, 366.6666666666667, 1000.0, 150.0, 388.8888888888889, 438.8888888888889],
])

# alpha = 0.15, tol_alpha = 0.05
# N = 8000, init 40
samples_4d = np.array([
    [204.25, 429.0, 312.5, 150.0, 287.5, 412.5]
])

fpr_2d = np.array([
    [0.0058823529411764705, 0.017993079584775088, 0.0038062283737024223, 0.010899653979238755, 0.08823529411764706, 0.03944636678200692],
    [0.02784653465346535, 0.005775577557755776, 0.020214521452145216, 0.026815181518151814, 0.2838283828382838, 0.24628712871287128],
    [0.014705882352941175, 0.017993079584775088, 0.0038062283737024223, 0.010899653979238755, 0.08823529411764706, 0.03944636678200692],
    [0.0627062706270627, 0.0066006600660066, 0.018151815181518153, 0.025165016501650164, 0.2838283828382838, 0.25165016501650167],
])

fpr_3d = np.array([
    [0.005834985133795838, 0.005271308225966302, 0.000715436075322101, 0.004630203171456888, 0.0674554013875124, 0.029980178394449948],
    [0.03391784641784642, 0.0015015015015015015, 0.0016623766623766623, 0.0016623766623766623, 0.27193264693264696, 0.1532961532961533],
    [0.006569348089417466, 0.0054853540358991305, 0.000739868957163308, 0.005089610175090849, 0.0674554013875124, 0.030616809822706745],
])

fpr_4d = np.array([
    [0.007185210835201554, 0.000712184603961103, 0.005722465765161143, 0.0063900272736359615, 0.2417500821064957, 0.09342380517199526],
])

fnr_2d = np.array([
    [0.016782334384858044, 0.06971608832807571, 0.1583596214511041, 0.09911671924290222, 0.0031545741324921135, 0.019558359621451103],
    [0.04447655748233783, 0.6500481695568401, 0.40815671162491973, 0.35589274245343616, 0.010918432883750802, 0.023603082851637765],
    [0.023280757097791795, 0.06971608832807571, 0.1583596214511041, 0.09911671924290222, 0.0031545741324921135, 0.019558359621451103],
    [0.051541425818882464, 0.6454720616570327, 0.426461143224149, 0.3683365446371227, 0.010918432883750802, 0.021355170199100833],
])

fnr_3d = np.array([
    [0.08042317287258617, 0.14494872031784162, 0.23732606486186825, 0.14315069758846902, 0.004952416150789984, 0.027620807539499213],
    [0.16738957786059017, 0.5796661711500736, 0.5493514162950293, 0.5493514162950293, 0.011281487364326876, 0.049178324882399656],
    [0.10356545217489503, 0.14628928105782948, 0.23616988512119252, 0.13153058815074894, 0.004952416150789984, 0.026870758775036703],
])

fnr_4d = np.array([
    [0.3397566589282799, 0.7424849380199021, 0.4208593989543223, 0.39830823998821324, 0.005690302593487809, 0.04526358962923438],
])

# Success counts
successes_2d = np.array([
    [10, np.nan, np.nan, np.nan, np.nan, np.nan],
    [8, np.nan, np.nan, np.nan, np.nan, np.nan],
    [10, np.nan, np.nan, np.nan, np.nan, np.nan],
    [4, np.nan, np.nan, np.nan, np.nan, np.nan],
])

successes_3d = np.array([
    [10, np.nan, np.nan, np.nan, np.nan, np.nan],
    [3, np.nan, np.nan, np.nan, np.nan, np.nan],
    [9, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0, np.nan, np.nan, np.nan, np.nan, np.nan],
])

successes_4d = np.array([
    [4, np.nan, np.nan, np.nan, np.nan, np.nan],
])

# -----------------------------
# Function to plot Number of Samples
# -----------------------------
def plot_samples(samples, dim_name, successes, figsize=(8,5)):
    x = np.arange(len(samples))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=figsize)
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
        for c in range(len(samples)):
            if not np.isnan(successes[c,i]):
                labels.append(f"{successes[c,i]}/10")
            else:
                labels.append("")
        # labels = [f"{successes[c,i]}/10" for c in range(num_conditions)]
        # ax.bar_label(bars, labels=labels, padding=2, fontsize=8)
        for bar, label in zip(bars, labels):
            height = bar.get_height()
            if label is None or height is None or np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=8
            )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions[:len(samples)], rotation=30)
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Number of Samples ({dim_name})")
    # ax.legend(loc='upper center', ncol=num_methods, bbox_to_anchor=(0.5, 1.15))
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Function to plot FPR/FNR combined
# -----------------------------
def plot_fpr_fnr(fpr, fnr, dim_name, successes, figsize=(8,5)):
    x = np.arange(len(fpr))
    width = 0.07
    
    fig, ax = plt.subplots(figsize=figsize)
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
        for c in range(len(samples)):
            if not np.isnan(successes[c,i]):
                labels.append(f"{successes[c,i]}/10")
            else:
                labels.append("")
        # labels = [f"{successes[c,i]}/10" for c in range(num_conditions)]
        # ax.bar_label(bars_fpr, labels=labels, padding=2, fontsize=8)
        # ax.bar_label(bars_fnr, labels=labels, padding=2, fontsize=8)
        for bar, label in zip(bars_fpr, labels):
            height = bar.get_height()
            if label is None or height is None or np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=8
            )
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions[:len(samples)], rotation=30)
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
dimensions = [("2D", samples_2d, fpr_2d, fnr_2d, successes_2d),
              ("3D", samples_3d, fpr_3d, fnr_3d, successes_3d),
              ("4D", samples_4d, fpr_4d, fnr_4d, successes_4d)]

for dim_name, samples, fpr, fnr, successes in dimensions:
    if dim_name == "4D":
        plot_samples(samples, dim_name, successes, (5,5))
        plot_fpr_fnr(fpr, fnr, dim_name, successes, (6,5))
    else:
        plot_samples(samples, dim_name, successes)
        plot_fpr_fnr(fpr, fnr, dim_name, successes)