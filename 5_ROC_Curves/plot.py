import matplotlib.pyplot as plt

import numpy as np

import os, sys, pickle

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from rcparams import plotter, textwidth, columnwidth

@plotter()
def main():

    with open('CNN_eval_v2.pkl', 'rb') as f:
        data = pickle.load(f)
        fpr = data['fpr']
        tpr = data['tpr']
        f1s = data['f1s']
        thresholds = data['thresholds_pr']
        best_thresh = data['best_thresh']

    print(data.keys())

    fig, axes = plt.subplots(figsize=(textwidth,textwidth*0.437), nrows=1, ncols=2, dpi=150)
    
    ax = axes[0]
    ix = (tpr - fpr).argmax() if hasattr(tpr, 'argmax') else (tpr - fpr).index(max(tpr - fpr))

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
    ax.plot(fpr, tpr, color='k', label=f'CNN (AUC=0.99)', zorder=1)
    ax.scatter(fpr[ix], tpr[ix], s=10, color='crimson', label='Threshold', zorder=2)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax = axes[1]

    ax.axvline(best_thresh, color='crimson', linestyle='--', alpha=1)
    ax.plot(thresholds, f1s, color='k')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-Score')

    # collect handles & labels from all axes
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # now make the legend only for items with labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.875)
    plt.savefig(f'{os.path.basename(os.getcwd())}.png', dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
