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
    
    with open('../../denoise_data21_v2.pkl', 'rb') as f:
        df21 = pickle.load(f)

    print(df21.columns)

    i=6747
    recon_pulse = df21.iloc[i]['recon_pulse']
    recon_deriv = df21['original_deriv_norm'].iloc[i]
    pulse = Pulse(recon_pulse)
    trise = pulse.find_rise_time() # trise has 3 arguments (total time, intital time, final time)

    *tfwhm, fwhmNorm = pulse.find_peak_fwhm_time(type='fwhm', return_norm=True) # tfwhm has the same arguments as trise

    fig, axes = plt.subplots(figsize=(textwidth,textwidth*0.4), nrows=1, ncols=2, dpi=130)

    ax = axes[0]
    ax.plot(pulse.time, pulse.data, color='black')

    rise_mask = (pulse.time >= trise[1]) & (pulse.time <= trise[2])
    fwhm_mask = (pulse.time >= tfwhm[1]) & (pulse.time <= tfwhm[2])
    
    ax.plot(pulse.time[rise_mask], pulse.data[rise_mask], color='red', linewidth=2, label='Rise Time')
    # ax.plot(pulse.time[fwhm_mask], pulse.data[fwhm_mask], color='green', linewidth=2, label='FWHM Time')

    ax=axes[1]
    ax.plot(pulse.time, recon_deriv, color='black')
    ax.plot(pulse.time[fwhm_mask], recon_deriv[fwhm_mask], color='green', linewidth=2, label='FWHM Time')

    i_max = np.argmax(recon_deriv)
    ax.plot(pulse.time[i_max], recon_deriv[i_max], 'rx', label='Detected Peak')
    
    for ax in axes:
        ax.set_ylabel('Amplitude [A.U.]')
        ax.set_xlabel('Time [µs]')
        ax.set_xlim(0, pulse.time[i_max]*2)
        ax.legend(loc='upper left')
    
    print(pulse.time[-1])

    plt.tight_layout()
    plt.savefig(f'{os.path.basename(os.getcwd())}.png',  bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
