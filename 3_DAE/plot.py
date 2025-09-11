import matplotlib.pyplot as plt

import numpy as np

import os, sys

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from rcparams import plotter, textwidth, columnwidth

@plotter()
def main():
    
    fig, ax = plt.subplots(figsize=(textwidth,textwidth*0.6), nrows=1, ncols=1, dpi=130)

    ax.plot()

    ax.set_xlim()
    ax.set_xlabel('')
    ax.set_ylim()
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(f'{os.path.basename(os.getcwd())}.png',  bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
