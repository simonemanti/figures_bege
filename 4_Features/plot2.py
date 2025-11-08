import matplotlib.pyplot as plt
import numpy as np
import os, sys, pickle
import pandas as pd

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
    
from pulse import Pulse
from pulse_batch import PulseBatch
from rcparams import plotter, textwidth, columnwidth

@plotter()
def main():
    
    with open('../../processed_data/2021/denoise_data21_v2.pkl', 'rb') as f:
        df21 = pickle.load(f)

    # =============== Rise Time ===============
    pulses_data21 = np.stack(df21['original_pulse'])
    pulses21 = PulseBatch(pulses_data21)
    rise_times = pulses21.find_rise_time()[0]

    nbins = 100
    xlower = 0
    xupper = 1
    xbins = np.linspace(xlower, xupper, nbins)
    
    # Calculate statistics
    rise_times_recon = df21['rise_time']
    mu = np.nanmean(rise_times_recon)
    sigma = np.nanstd(rise_times_recon)
    
    # Plot histogram

    plt.figure(figsize=(textwidth,textwidth*0.6), dpi=130)
    plt.hist(rise_times, bins=xbins, color='grey', alpha=0.7,label='before')
    plt.hist(rise_times_recon, bins=xbins, histtype='step', color='brown')
    
    # Fill ±1σ and ±2σ regions
    plt.axvspan(mu-2*sigma, mu+2*sigma, color='yellow', alpha=0.2, label='±2σ')
    plt.axvspan(mu-sigma, mu+sigma, color='blue', alpha=0.2, label='±1σ')
    
    # Draw mean and std lines
    plt.axvline(mu, color='red', linestyle='--', label=f'Mean {mu:.3f}')
    plt.axvline(mu-sigma, color='blue', linestyle=':', label=f'Mean -1σ {mu-sigma:.3f}')
    plt.axvline(mu+sigma, color='blue', linestyle=':', label=f'Mean +1σ {mu+sigma:.3f}')
    plt.axvline(mu-2*sigma, color='gold', linestyle=':', label=f'Mean -2σ {mu+2*sigma:.3f}')
    plt.axvline(mu+2*sigma, color='gold', linestyle=':', label=f'Mean +2σ {mu-2*sigma:.3f}')
    plt.xlim(0)
    
    plt.xlabel('Rise Time (μs)')
    plt.ylabel(f'Count/{(xupper-xlower)/nbins:.3f}μs')
    plt.yscale('log')
    plt.title(f'Rise Time Before v.s. After Denoising')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{os.path.basename(os.getcwd())}_riseTimeDistribution.png',  bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}_riseTimeDistribution.pdf')
    plt.show()

    # =============== FWHM Time ===============
    *tsfwhm21, _ = pulses21.find_peak_fwhm_time(type='fwhm')
    fwhm_times = tsfwhm21[0]
    
    recon_pulses_data21 = np.stack(df21['recon_pulse'])
    pulses21_recon = PulseBatch(recon_pulses_data21)
    *tsfwhm21_recon, _ = pulses21_recon.find_peak_fwhm_time(type='fwhm')
    fwhm_times_recon = tsfwhm21_recon[0]

    nbins = 100
    xlower = 0
    xupper = 2.25
    xbins = np.linspace(xlower, xupper, nbins)
    
    mask = fwhm_times!=999
    masks_recon = fwhm_times_recon!=999
    # Calculate statistics
    mu = np.nanmean(fwhm_times_recon[masks_recon])
    sigma = np.nanstd(fwhm_times_recon[masks_recon])
    
    # Plot histogram
    plt.figure(figsize=(textwidth,textwidth*0.6), dpi=130)
    plt.hist(fwhm_times[mask], bins=xbins, color='grey', alpha=0.7,label='before')
    plt.hist(fwhm_times_recon[masks_recon], bins=xbins, histtype='step', color='brown',label='after')
    
    # Fill ±1σ and ±2σ regions
    plt.axvspan(mu-2*sigma, mu+2*sigma, color='yellow', alpha=0.2, label='±2σ')
    plt.axvspan(mu-sigma, mu+sigma, color='blue', alpha=0.2, label='±1σ')
    
    # Draw mean and std lines
    plt.axvline(mu, color='red', linestyle='--', label=f'Mean {mu:.3f}')
    plt.axvline(mu-sigma, color='blue', linestyle=':', label=f'Mean -1σ {mu-sigma:.3f}')
    plt.axvline(mu+sigma, color='blue', linestyle=':', label=f'Mean +1σ {mu+sigma:.3f}')
    plt.axvline(mu-2*sigma, color='gold', linestyle=':', label=f'Mean -2σ {mu+2*sigma:.3f}')
    plt.axvline(mu+2*sigma, color='gold', linestyle=':', label=f'Mean +2σ {mu-2*sigma:.3f}')
    plt.xlim(0)
    
    plt.xlabel('FWHM Time (μs)')
    plt.ylabel(f'Count/{(xupper-xlower)/nbins:.3f}μs')
    plt.yscale('log')
    plt.title(f'FWHM Time Before v.s. After Denoising')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{os.path.basename(os.getcwd())}_fwhmTimeDistribution.png',  bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}_fwhmTimeDistribution.pdf')
    plt.show()



if __name__ == '__main__':
    main()
