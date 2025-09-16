import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pickle
import pandas as pd

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from pulse_batch import PulseGenerator, PulseBatch
from rcparams import plotter, textwidth, columnwidth

@plotter()
def main():
    
    n_channels = 1024
    sampling_rate = 400e6  # Hz
    dt = 1 / sampling_rate  # seconds per channel/sample
    t_n = np.arange(n_channels) * dt * 1e6

    gen = PulseGenerator()
    slow_rises, _ = gen.simulate_pulses(N_PULSES=10, N_SAMPLES=1024,
                                        PROB_NORMAL=0, PROB_DOUBLE_STEP=0,PROB_FLAT_TOP=0,PROB_SLOW_RISER=1)
    nominal_pulses, _ = gen.simulate_pulses(N_PULSES=10, N_SAMPLES=1024,
                                            PROB_NORMAL=1, PROB_DOUBLE_STEP=0,PROB_FLAT_TOP=0,PROB_SLOW_RISER=0)
    pile_ups, _ = gen.simulate_pulses(N_PULSES=10, N_SAMPLES=1024,
                                      PROB_NORMAL=0, PROB_DOUBLE_STEP=1,PROB_FLAT_TOP=0,PROB_SLOW_RISER=0)
    flat_tops, _ = gen.simulate_pulses(N_PULSES=10, N_SAMPLES=1024,
                                       PROB_NORMAL=0, PROB_DOUBLE_STEP=0,PROB_FLAT_TOP=1,PROB_SLOW_RISER=0)

    slow_rise = slow_rises[np.random.choice(len(slow_rises))]
    nominal_pulse = nominal_pulses[np.random.choice(len(nominal_pulses))]
    pile_up = pile_ups[np.random.choice(len(pile_ups))]
    flat_top = flat_tops[np.random.choice(len(flat_tops))]
    
    fig, axes = plt.subplots(figsize=(textwidth,textwidth*0.6), nrows=2, ncols=2, dpi=130, sharex=True)

    ax = axes[0,0]
    ax.plot(t_n, nominal_pulse, color='black')
    ax.set_ylabel('Amplitude (A.U.)')
    ax.set_xlabel('Time (µs)')

    ax = axes[0,1]
    ax.plot(t_n, slow_rise, color='black')

    ax = axes[1,0]
    ax.plot(t_n, pile_up, color='black')

    ax = axes[1,1]
    ax.plot(t_n, flat_top, color='black')

    plt.tight_layout()
    plt.savefig(f'{os.path.basename(os.getcwd())}.png',  bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf')
    plt.show()


if __name__ == '__main__':
    main()
