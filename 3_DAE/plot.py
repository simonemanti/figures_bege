import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pickle
import pandas as pd
p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from pulse import Pulse
from rcparams import plotter, textwidth, columnwidth

@plotter()
def main():
    n_channels = 1024
    sampling_rate = 400e6
    dt = 1 / sampling_rate
    t_n = np.arange(n_channels) * dt * 1e6
    
    with open('../../denoise_data21_v6.pkl', 'rb') as f:
        df21 = pickle.load(f)
    
    i = 6747
    ori_pulse = df21['original_normalized_pulse'].iloc[i]
    recon = df21['recon_pulse'].iloc[i]
    recon_deriv = df21['original_deriv_norm'].iloc[i]
    
    fig, axes = plt.subplots(figsize=(textwidth, textwidth*0.437), nrows=1, ncols=2, dpi=130)
    
    axes[0].plot(t_n[:-100], ori_pulse, label='Original', c='C0')
    axes[0].plot(t_n[:-100], recon, label='Reconstructed', color='C1')
    axes[0].grid(ls=':')
    
    axes[1].plot(t_n[:-100], Pulse(ori_pulse).normalize_deriv(), c='C0')
    axes[1].plot(t_n[:-100], recon_deriv, color='C1')
    axes[1].grid(ls=':')
    
    for ax in axes:
        ax.set_ylabel('Amplitude [A.U.]')
        ax.set_xlabel('Time [µs]')
        ax.set_xlim(0, 2.5)
    
    fig.legend(['Original', 'Reconstructed'], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.savefig(f'{os.path.basename(os.getcwd())}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf')
    plt.show()

if __name__ == '__main__':
    main()