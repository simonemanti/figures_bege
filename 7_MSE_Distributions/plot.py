import matplotlib.pyplot as plt
import numpy as np
import os, sys, pickle
import pandas as pd

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
    
from pulse import Pulse
from rcparams import plotter, textwidth, columnwidth

@plotter()
def main():
    
    with open("../../processed_data/2021/DAE_mse_sampleVSdata.pkl", "rb") as f:
        y_pred, Y_val_clean_norm, norm21, recon21 = pickle.load(f)


    plt.figure(figsize=(textwidth,textwidth*0.6))

    recon_error_data = np.mean((norm21 - recon21) ** 2, axis=1)
    recon_error = np.mean((Y_val_clean_norm - y_pred) ** 2, axis=1)
    
    low, high, bins = 0, 0.0006, 100
    bins = np.linspace(low,high, bins)
    
    plt.hist(recon_error_data, bins=bins, histtype='step', label='Data', density=True)
    plt.hist(recon_error, bins=bins, histtype='step', label='Training', density=True)
    plt.legend()
    plt.ylabel('Counts')
    plt.xlabel('MSE')
    plt.xlim(low,high)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{os.path.basename(os.getcwd())}.png',  bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
