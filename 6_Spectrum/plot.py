import matplotlib.pyplot as plt
import numpy as np
import os, sys, pickle
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from rcparams import plotter, textwidth, columnwidth

@plotter()
def main():
    # Load data
    with open('../../data21_with_label_v2.pkl', 'rb') as f:
        df = pickle.load(f)
    
    mask = (df['pred_label']==1)
    spectrum_total = df['energy'].values
    spectrum_good = df[mask]['energy'].values
    spectrum_bad = df[~mask]['energy'].values
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(textwidth,textwidth*0.437), dpi=150)
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.0)
    
    # Left: spectrum comparison
    ax1 = fig.add_subplot(gs[:, 0])
    plot_spectrum_comparison(ax1, spectrum_total, spectrum_good, spectrum_bad)
    
    # Right top: fitted spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    plot_fitted_spectrum_with_pull(ax2, ax3, spectrum_good)
    
    plt.tight_layout()
    plt.savefig(f'{os.path.basename(os.getcwd())}_analysis_2.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{os.path.basename(os.getcwd())}_analysis_2.pdf', bbox_inches='tight')
    plt.show()

def plot_spectrum_comparison(ax, total, good, bad):
    """Plot comparison of total, good, and bad spectra"""
    xbins = np.linspace(0, 1000, 500)

    lw = 0.4
    
    ax.hist(total, bins=xbins, histtype='step', linewidth=lw, color='k')
    ax.hist(good, bins=xbins, histtype='step', linewidth=lw, color='C2')
    ax.hist(bad, bins=xbins, histtype='step', linewidth=lw, color='r')

    ax.plot([],[], c='k', label='Total')
    ax.plot([],[], c='g', label='Good')
    ax.plot([],[], c='r', label='Bad')

    ax.set_xlim(0, 650)
    ax.set_ylim(0, 200)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Counts / 1 keV')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

def plot_fitted_spectrum_with_pull(ax_main, ax_pull, spectrum):
    """Plot spectrum with fit and pull plot below"""
    # Create histogram
    counts, edges = np.histogram(spectrum, bins=500, range=(0, 1000))
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Fit range
    lower, upper = 15, 650
    fit_mask = (centers >= lower) & (centers <= upper)
    x_fit = centers[fit_mask]
    y_fit = counts[fit_mask]
    
    # Perform fit
    params, cov, peak_info = fit_spectrum(x_fit, y_fit)
    
    # Main plot
    # ax_main.step(centers[:650], counts[:650], 'k-', lw=0.8, label='Data', where='mid')
    ax_main.errorbar(centers[:650], counts[:650], yerr=np.sqrt(counts[:650]), xerr=np.diff(centers)[0]/2, fmt='o', ms=0, lw=0.3, color='k', label='Data', zorder=3)
    
    x_smooth = np.linspace(lower, upper, 5000)
    y_bg = background_model(x_smooth, *params[:8])
    # ax_main.plot(x_smooth, y_bg, 'gray', lw=0.5, alpha=1, label='Background', zorder=2)

    y_total = multi_peak_model(x_smooth, *params)
    ax_main.plot(x_smooth, y_total, 'r-', lw=0.7, label='Fit', zorder=4)
    
    plot_individual_peaks(ax_main, x_smooth, params, peak_info)
    # add_peak_annotations(ax_main, params, cov, peak_info)
    
    ax_main.set_xlim(0, 650)
    ax_main.set_ylim(0, 200)
    ax_main.set_ylabel('Counts / 1 keV')
    ax_main.legend(loc='upper right')
    ax_main.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax_main.yaxis.set_major_locator(ticker.MultipleLocator(50))
    # ax_main.set_xticklabels([])
    
    # Pull plot
    y_fit_eval = multi_peak_model(centers[:650], *params)
    errors = np.sqrt(counts[:650] + 1)  # Poisson errors
    pull = np.where(errors > 0, (counts[:650] - y_fit_eval) / errors, 0)
    
    # Green region (-1, 1)
    ax_pull.axhspan(-1, 1, facecolor='greenyellow', alpha=1)

    # Yellow regions (-2, -1) and (1, 2)
    ax_pull.axhspan(-2, -1, facecolor='yellow', alpha=1)
    ax_pull.axhspan(1, 2, facecolor='yellow', alpha=1)

    ax_pull.scatter(centers[:650], pull, s=0.3, c='k', alpha=1)
    # ax_pull.axhline(0, color='r', lw=1.5)
    # ax_pull.axhline(3, color='gray', lw=0.5, ls='--')
    # ax_pull.axhline(-3, color='gray', lw=0.5, ls='--')
    
    ax_pull.set_xlim(0, 650)
    ax_pull.set_yticks([-2,0,2])
    ax_pull.set_ylim(-4, 4)
    ax_pull.set_xlabel('Energy [keV]')
    ax_pull.set_ylabel('Pull')
    # ax_pull.grid(ls=':', axis='y')
    ax_pull.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # ax_pull.tick_params(axis='y')

def background_model(x, *params):
    """Double exponential with error functions"""
    a1, b1, c1, d1, a2, b2, c2, d2 = params
    bg1 = np.exp(a1 + b1*x) * 0.5 * (1 + erf((x - c1)/d1))
    bg2 = np.exp(a2 + b2*x) * 0.5 * (1 + erf((x - c2)/d2))
    return bg1 + bg2

def gaussian(x, amp, mu, sigma):
    """Gaussian function"""
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def multi_peak_model(x, *params):
    """Combined model: background + 7 Gaussian peaks"""
    y = background_model(x, *params[:8])
    for i in range(7):
        idx = 8 + i*3
        y += gaussian(x, params[idx], params[idx+1], params[idx+2])
    return y

def fit_spectrum(x, y):
    """Fit spectrum with background and peaks"""
    p0 = get_initial_params()
    bounds = get_param_bounds()
    
    peak_info = [
        ('$^{210}$Pb', 25), ('', 45), ('$^{212}$Pb', 74),
        ('$^{214}$Pb', 237), ('$^{214}$Pb', 292),
        ('$^{214}$Bi', 348), ('', 605)
    ]
    
    try:
        params, cov = curve_fit(multi_peak_model, x, y, p0=p0, bounds=bounds, maxfev=5000)
    except:
        params, cov = p0, np.eye(len(p0))
    
    return params, cov, peak_info

def get_initial_params():
    """Get initial parameters for fit"""
    bg_params = [1, -0.01, 100, 30, 1, -0.01, 300, 30]
    peak_params = [
        40, 25, 1.5,      # 210-Pb
        150, 45, 0.3,     # peak 2
        37, 74, 0.3,      # 212-Pb
        50, 237, 2.3,     # 214-Pb
        40, 292, 0.3,     # 214-Pb
        42, 348, 0.3,     # 214-Bi
        27, 605, 0.3      # peak 7
    ]
    return bg_params + peak_params

def get_param_bounds():
    """Get parameter bounds for fit"""
    bg_lower = [0, -1, 80, 10, 0, -1, 200, 10]
    bg_upper = [10, 0, 130, 200, 10, 0, 350, 200]
    peak_lower = [
        35, 23, 0.1,    150, 44, 0.1,    15, 72, 0.1,
        25, 234, 0.1,   20, 291, 0.1,    30, 347, 0.1,
        25, 603, 0.1
    ]
    peak_upper = [
        50, 28, 5,      200, 46, 3,      50, 76, 3,
        70, 240, 3,     100, 295, 3,     70, 350, 1,
        50, 608, 1
    ]
    return (bg_lower + peak_lower, bg_upper + peak_upper)

def plot_individual_peaks(ax, x, params, peak_info):
    """Plot individual Gaussian peaks"""
    colors = ['C0','C1','C2','C3','C4','C5','C6']
    for i, (name, color) in enumerate(zip(peak_info, colors)):
        idx = 8 + i*3
        y_peak = gaussian(x, params[idx], params[idx+1], params[idx+2])
        if name[0]:
            ax.plot(x, y_peak, '-', color=color, lw=0.5, alpha=1, zorder=1)

def add_peak_annotations(ax, params, cov, peak_info):
    """Add peak labels and statistics"""
    idx = 8  # 210-Pb peak
    mu, sigma = params[idx+1], params[idx+2]
    
    x_min, x_max = mu - 2*sigma, mu + 2*sigma
    x_region = np.linspace(x_min, x_max, 100)
    
    S = gaussian(x_region, params[idx], mu, sigma).sum()
    B = background_model(x_region, *params[:8]).sum()
    significance = S / np.sqrt(B) if B > 0 else 0
    
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    
    # ax.text(0.95, 0.85, f'$^{{210}}$Pb FWHM = {fwhm:.2f} keV', 
            # transform=ax.transAxes, ha='right', fontsize=8)
    # ax.text(0.95, 0.80, f'$^{{210}}$Pb S/√B = {significance:.2f}',
            # transform=ax.transAxes, ha='right', fontsize=8)
    # ax.text(0.95, 0.90, 'VIP 2021', transform=ax.transAxes, 
            # ha='right', fontsize=8, style='italic')
    
    peak_positions = [(25, 0.9), (74, 0.5), (237, 0.55), 
                     (292, 0.55), (348, 0.4)]
    
    for (name, _), (x_pos, y_rel) in zip(peak_info, peak_positions):
        if name:
            y_pos = ax.get_ylim()[1] * y_rel
            ax.text(x_pos, y_pos, name, fontsize=7, ha='center')

if __name__ == '__main__':
    main()