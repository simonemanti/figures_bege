import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import h5py
import os, sys
p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from pulse import Pulse
from rcparams import plotter, textwidth, columnwidth

@plotter()
def main():
    # Load data
    with h5py.File('data_plot.h5', 'r') as f:
        rise_times = f['rise_times'][:]
        rise_times_recon = f['rise_times_recon'][:]
        fwhm_times = f['fwhm_times'][:]
        fwhm_times_recon = f['fwhm_times_recon'][:]
    
    with h5py.File('data_mse.h5', 'r') as f:
        recon_error_data = f['recon_error_data'][:]
        recon_error = f['recon_error'][:]
    
    fig, axes = plt.subplots(1, 3, figsize=(textwidth*3/2, textwidth*0.437), dpi=150)
    
    # Subplot 1: Rise Time
    ax = axes[0]
    xbins = np.linspace(0, 1, 100)
    mu = np.nanmean(rise_times_recon)
    sigma = np.nanstd(rise_times_recon)
    
    ax.hist(rise_times, bins=xbins, color='grey', alpha=0.7, label='Before')
    ax.hist(rise_times_recon, bins=xbins, histtype='step', color='brown', label='After')
    ax.axvspan(mu-2*sigma, mu+2*sigma, color='yellow', alpha=0.2, label='±2σ')
    ax.axvspan(mu-sigma, mu+sigma, color='greenyellow', alpha=0.2, label='±1σ')
    ax.axvline(mu, color='red', linestyle='--', label=f'Mean')
    # ax.axvline(mu-sigma, color='blue', linestyle=':', label=f'Mean -1σ {mu-sigma:.3f}')
    # ax.axvline(mu+sigma, color='blue', linestyle=':', label=f'Mean +1σ {mu+sigma:.3f}')
    # ax.axvline(mu-2*sigma, color='gold', linestyle=':', label=f'Mean -2σ {mu+2*sigma:.3f}')
    # ax.axvline(mu+2*sigma, color='gold', linestyle=':', label=f'Mean +2σ {mu-2*sigma:.3f}')
    ax.set_xlim(0,1)
    ax.set_xlabel('Rise Time [μs]')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_title('Rise Time Before v.s. After Denoising')
    ax.legend()
    ax.grid(ls=':')
    
    # Subplot 2: FWHM Time
    ax = axes[1]
    xbins = np.linspace(0, 2.25, 100)
    mask = fwhm_times != 999
    masks_recon = fwhm_times_recon != 999
    mu = np.nanmean(fwhm_times_recon[masks_recon])
    sigma = np.nanstd(fwhm_times_recon[masks_recon])
    
    ax.hist(fwhm_times[mask], bins=xbins, color='grey', alpha=0.7, label='Before')
    ax.hist(fwhm_times_recon[masks_recon], bins=xbins, histtype='step', color='brown', label='After')
    ax.axvspan(mu-2*sigma, mu+2*sigma, color='yellow', alpha=0.2, label='±2σ')
    ax.axvspan(mu-sigma, mu+sigma, color='greenyellow', alpha=0.2, label='±1σ')
    ax.axvline(mu, color='red', linestyle='--', label=f'Mean')
    # ax.axvline(mu-sigma, color='blue', linestyle=':', label=f'Mean -1σ {mu-sigma:.3f}')
    # ax.axvline(mu+sigma, color='blue', linestyle=':', label=f'Mean +1σ {mu+sigma:.3f}')
    # ax.axvline(mu-2*sigma, color='gold', linestyle=':', label=f'Mean -2σ {mu+2*sigma:.3f}')
    # ax.axvline(mu+2*sigma, color='gold', linestyle=':', label=f'Mean +2σ {mu-2*sigma:.3f}')
    ax.set_xlim(0,2)
    ax.set_xlabel('FWHM Time [μs]')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_title('FWHM Time Before v.s. After Denoising')
    ax.legend()
    ax.grid(ls=':')
    
    # Subplot 3: MSE
    ax = axes[2]
    bins = np.linspace(0, 0.0006, 100)
    ax.hist(recon_error_data, bins=bins, histtype='step', label='Real', density=True)
    ax.hist(recon_error, bins=bins, histtype='step', label='Synthetic', density=True)
    ax.set_xlabel('MSE')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_xlim(0, 0.0005)
    ax.set_ylim(1e1, 1e5)
    ax.legend()
    ax.grid(ls=':')
    
    plt.tight_layout()
    plt.savefig(f'{os.path.basename(os.getcwd())}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf')
    plt.show()

if __name__ == '__main__':
    main()
