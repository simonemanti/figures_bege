import numpy as np
import os, sys, pickle
import h5py

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
    
from pulse import Pulse
from pulse_batch import PulseBatch

def main():
    with open('../../denoise_data21_v2.pkl', 'rb') as f:
        df21 = pickle.load(f)

    # Rise Time
    pulses_data21 = np.stack(df21['original_pulse'])
    pulses21 = PulseBatch(pulses_data21)
    rise_times = pulses21.find_rise_time()[0]
    rise_times_recon = df21['rise_time']

    # FWHM Time
    *tsfwhm21, _ = pulses21.find_peak_fwhm_time(type='fwhm')
    fwhm_times = tsfwhm21[0]
    
    recon_pulses_data21 = np.stack(df21['recon_pulse'])
    pulses21_recon = PulseBatch(recon_pulses_data21)
    *tsfwhm21_recon, _ = pulses21_recon.find_peak_fwhm_time(type='fwhm')
    fwhm_times_recon = tsfwhm21_recon[0]

    # Save to HDF5
    with h5py.File('data_plot.h5', 'w') as f:
        f.create_dataset('rise_times', data=rise_times)
        f.create_dataset('rise_times_recon', data=rise_times_recon)
        f.create_dataset('fwhm_times', data=fwhm_times)
        f.create_dataset('fwhm_times_recon', data=fwhm_times_recon)

if __name__ == '__main__':
    main()