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
    
    fig, ax = plt.subplots(figsize=(textwidth,textwidth*0.6), nrows=1, ncols=1, dpi=130)
    
    ix = (tpr - fpr).argmax() if hasattr(tpr, 'argmax') else (tpr - fpr).index(max(tpr - fpr))
    ax.plot(fpr, tpr, color='black', label=f'ROC curve (AUC = 0.999)')
    ax.scatter(fpr[ix], tpr[ix], s=50, label='Threshold')

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{os.path.basename(os.getcwd())}.png',  bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
