""" SATIS TOOL

This tool performs basic spectral analysis, such as Fourier Transform and
Power Spectral Density.

It works on one or several arrays.

Special emphasis is put on:
- Truncating the time support so as to match exactly a target frequency in
the frequency content of the spectral transforms.
- Checking the convergence with the length of the time support.
- Checking the variability among the signals.
- Checking that the Parseval equality is respected.
"""
import sys
import argparse
from textwrap import wrap
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from satis.satis_library import (get_average_psd, ROUND_PRECISION, define_good_time_array, interpolate_signals,
                       define_good_time_array, get_cumulative_average, get_coeff_fourier,
                       get_clean_fft, compute_cumulative_fourier, get_psd_harmonics)
from satis.cerfacs_plot import add_credits, tight_plot

import pandas as pd

__all__ = ["display_header", "extract_subset", "read_param", "read_signals", "read_signal_yaml", "clean_signals",
           "set_default_target_freq", "plot_time_signals", "plot_fourier_convergence", "plot_psd_convergence",
           "plot_fourier_variability", "plot_psd_variability", "write_output"]

ROUND_PRECISION = ROUND_PRECISION


def _wrap_title(title, length=40):
    """ 
    *Wrap text*
    
    :param title
    :type title: string
    :param length
    :type length: int

    :returns:
        - Your title clipped at *length* characters

     """
    return "\n".join(wrap(title, length))


# TODO : find better name ?
def _my_angle(complex_array):
    """ 
    *Return the angles of arrays, between 0 and 2pi*

    :param complex_array
    :type complex_array: array

    :returns:
        - **angles** - Array of angles normalized between 0 and 2pi
     """
    angles = np.mod(np.angle(complex_array), 2 * np.pi)
    return angles


def display_header():
    """ Display the tool header """
    print("-------------------------------------------------------")
    print("                         SATIS                         ")
    print("-------------------------------------------------------")
    return


def extract_subset(subset, signals):
    """ 
    *Extract only a subset of signals, referenced by the list indices.*

    :param subset 
    :type subset: list
    :param signals 
    :type signals: array

    :returns:
        - **signals** - Array with selected signals
    """
    print(("--> Extracting signals: {0}".format(", ".join(map(str,subset)))))
    print(" ")
    return signals[subset, :]


def read_param(inargs):
    """
    *Parse the options (arguments) specified when calling avbp2global.py*

    :param inargs
    :type inargs: list

    :returns:
        - **outargs** - Parsed arguments

    """

    # TODO find an alternative to option parsing as the number options
    # keeps growing

    # Initialize parser
    parser = argparse.ArgumentParser(
        description=('Performs spectral analysis (FFT or PSD) and'
                     ' provides a few quality diagnostics'),
        epilog='Example: TODO')

    # Filename (containing all signals)
    # This file should contain N+1 columns with
    # Col.1 = Time
    # Col.2 to N+1 = Signal
    parser.add_argument('filename',
                        default='signals.dat',
                        type=str,
                        help='Path to the file containing the signals '
                             '(N+1 columns with time in col. 1). Default name is "signals.dat".')

    # Diagnostics
    # Five keywords available
    # time = plot temporal signals
    # fourier_convergence = plot Fourier transform of average signal
    #                              over complete, 1/2 and 1/4 signal
    # psd_convergence = same but for PSD
    # fourier_variability = Fourier coeff at target_frequency for all signals.
    # psd_variability = same but for PSD.
    diagnostics = ['time',
                   'fourier_convergence',
                   'psd_convergence',
                   'fourier_variability',
                   'psd_variability']
    parser.add_argument(
        'diagnostic',
        default='time',
        type=str,
        choices=diagnostics,
        help='Requested diagnostic. Default is "time".')

    # Discard Fourier phases for PSD computation
    # This is useful for the analysis of signals in phase opposition.
    parser.add_argument(
        '--discardphase',
        help='Set all Fourier phases to zero when computing the PSD.',
        action='store_true'
    )

    # Display PSD in db SPL
    parser.add_argument(
        '--dbspl',
        help='Display psd_convergence plot in dB SPL (valid only for pressure fluctuations).',
        action='store_true'
    )

    # Choose units
    parser.add_argument(
        '--unit',
        help='Unit of the input signals (for plots).',
        type=str,
        default="Unit"
    )

    # Target frequency
    parser.add_argument('-f',
                        '--freq',
                        default=0.0,
                        type=float,
                        required=False,
                        dest='target_frequency',
                        help='Set the target frequency in Hz.')

    # Initial transient time
    parser.add_argument('-t',
                        '--tinit',
                        default=0.0,
                        type=float,
                        required=False,
                        dest='transient_time',
                        help='Remove an initial portion of '
                             'the signal in s (optional). '
                             'Using 0.0 deactivates the filter.')

    # Windowing options
    parser.add_argument('--window_blocks',
                        default=1,
                        type=int,
                        required=False,
                        dest="nb_blocks",
                        help="For PSD convergence only: number of windows. Default is 1.")

    available_windows = ["flat",
                         "hanning",
                         "hamming",
                         "blackman",
                         "bartlett"]
    parser.add_argument('--window_shape',
                        default='flat',
                        type=str,
                        choices=available_windows,
                        dest="window_shape",
                        help='For PSD convergence only: shape of the windowing function. Default is flat.'
                        )

    parser.add_argument('--window_overlap',
                        default=0.0,
                        type=float,
                        dest="overlap",
                        help='For PSD convergence only: overlapping between consecutive windows. '
                             'Should be between 0 and 1. '
                             'Default is 0.0. '
                        )

    # Remove average option (probably removable)
    parser.add_argument('--remove_average',
        help='Remove temporal average of all signals.',
        action='store_true')

    # Extract subset 
    parser.add_argument('--subset',
        help='Remove temporal average of all signals.',
        type=int,
        nargs='*')

    # If no option is specified, return the parser help and quit program
    if len(inargs) == 1:
        parser.print_help()
        sys.exit(-1)
    else:
        # Otherwise, parse the arguments
        outargs = parser.parse_args()

    return outargs


def read_signals(filename, remove_average=False):
    """ 
    *Read the signals in file filename.*

    :param filename
    :type filename: string
    :param remove_average
    :type remove_average: bool

    :returns:

        - **time** - time vector
        - **signals** - array of signal values

    """
    
    # TODO: a complete shift from numpy to pandas is needed
    # data = np.loadtxt(filename)
    data = pd.read_table(filename, sep="\s+")
    time = data.iloc[:, 0].to_numpy()
    signals = data.iloc[:, 1:].transpose().to_numpy()
    print(" ")
    print(("--> Found {0} signals".format(signals.shape[0])))
    print(" ")
    if remove_average:
        signals -= np.mean(signals, axis=-1, keepdims=True)
    return time, signals


def read_signal_yaml(data):
    """ 
    *Extract time and signal, then convert them to arrays* 

    :param data
    :type data: yaml

    *returns*
        - **time_array** - Time vector
        - **signal_array** - Array of signal values

    """
    time = []
    signal = []
    signal_bis = []
    for item in data:
        time.append(item['time'])
        signal.append(item['ib_force_y'])
        signal_bis.append(item['ib_force_x'])
    time_array, signal_array = np.array(time), np.stack((np.array(signal), np.array(signal_bis)), axis=1)
    return time_array, np.transpose(signal_array)


def clean_signals(time, signals, target_frequency, transient_time, trim_end=False):
    """ 
    *Remove initial transient regime and truncated signals to fit
    in a whole number of periods*

    :param time
    :type time: array
    :param signals 
    :type signals: array
    :param target_frequency 
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param trim_end 
    :type trim_end: bool

    :returns:
        - **new_time** - Rescaled time vector
        - **new_signal** - Rescaled signals

    """

    # Truncate time array
    new_time = define_good_time_array(time,
                                         target_frequency,
                                         transient_time,
                                         trim_end)

    # Interpolate quantities on the clean time array
    new_signals = interpolate_signals(time, signals, new_time)

    return new_time, new_signals


def set_default_target_freq(time):
    """ 
    *Compute the Nyquist frequency and set the target frequency to a 3rd of it.*

    :param time
    :type time: array

    :returns:
        - **target_frequency** - Float equal to the third of the signals Nyquist frequency
    """
    timestep = np.mean(time[1:] - time[:-1])
    nyquist_freq = 0.5 / timestep
    target_frequency = nyquist_freq * 0.3

    return target_frequency


def plot_time_signals(time, signals, target_frequency, transient_time, unit):
    """ 
    *Plot temporal signals*
    :param time 
    :type time: array
    :param signals
    :type signals: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param unit
    :type unit: string

    :returns:
        - **figure** - The plot of the temporal signal
     """

    # Could be done with a decorator ?
    if target_frequency == 0.0:
        target_frequency = set_default_target_freq(time)

    new_time = define_good_time_array(time,
                                         target_frequency,
                                         transient_time)

    total_time = new_time[-1] + new_time[1] - 2 * new_time[0]
    # round is used to deal with precision problems
    nb_periods = int(round(total_time * target_frequency, ROUND_PRECISION))

    print("--- TIME DIAGNOSTICS ---")
    print(" ")
    print("--> Truncated time array")
    print(("    New initial time: {0:05.4f} s".format(new_time[0])))
    print(("    Number of periods: {0}".format(nb_periods)))
    print(("    Timestep: {0:05.4f} ms".format((time[1] - time[0]) * 1e3)))

    xmin = np.min(time)
    xmax = np.max(time)
    deltax = xmax - xmin

    ymin = np.min(signals)
    ymax = np.max(signals)
    deltay = ymax - ymin

    figure, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(7, 7))

    plt.xticks(rotation=70)

    # Plot all signals
    axes[0].plot(time, signals.transpose())
    rectangle1 = mpl.patches.Rectangle(xy=(xmin - 0.1 * deltax, ymin - 0.1 * deltay),
                                       width=new_time[0] -
                                             (xmin - 0.1 * deltax),
                                       height=ymax + 0.2 * deltay - ymin,
                                       color='grey',
                                       hatch='/',
                                       alpha=0.5)
    rectangle2 = mpl.patches.Rectangle(xy=(new_time[-1], ymin - 0.1 * deltay),
                                       width=xmax + 0.1 *
                                                    deltax - new_time[-1],
                                       height=ymax + 0.2 * deltay - ymin,
                                       color='grey',
                                       hatch='/',
                                       alpha=0.5)
    axes[0].add_patch(rectangle1)
    axes[0].add_patch(rectangle2)

    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude [{0}]'.format(unit))
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)
    xticks = np.arange(new_time[0], new_time[-1], 1. / target_frequency)
    axes[0].set_xticks(xticks)
    axes[0].grid(which='major', axis='x')
    axes[0].set_title('All signals')

    # Plot average signal
    average_signal = np.average(signals, axis=0)
    axes[1].plot(time, average_signal, color='black')
    rectangle1 = mpl.patches.Rectangle(xy=(xmin - 0.1 * deltax, ymin - 0.1 * deltay),
                                       width=new_time[0] -
                                             (xmin - 0.1 * deltax),
                                       height=ymax + 0.2 * deltay - ymin,
                                       color='grey',
                                       hatch='/',
                                       alpha=0.5)
    rectangle2 = mpl.patches.Rectangle(xy=(new_time[-1], ymin - 0.1 * deltay),
                                       width=xmax + 0.1 *
                                                    deltax - new_time[-1],
                                       height=ymax + 0.2 * deltay - ymin,
                                       color='grey',
                                       hatch='/',
                                       alpha=0.5)

    axes[1].add_patch(rectangle1)
    axes[1].add_patch(rectangle2)

    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Amplitude [{0}]'.format(unit))
    axes[1].set_xticks(xticks)
    axes[1].grid(which='major', axis='x')
    axes[1].set_title('Ensemble-average signal')

    # Plot cumulative average
    cumul_average = get_cumulative_average(new_time, average_signal, target_frequency, transient_time)
    cumul_average = np.append(cumul_average, cumul_average[-1])
    axes[2].step(np.arange(nb_periods + 1) / target_frequency + new_time[0], cumul_average, where='post')
    # axes[2].set_xticks(range(nb_periods))
    axes[2].grid(axis='x')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Amplitude [{0}]'.format(unit))
    axes[2].set_title('Cumulative time-average')

    # Show
    add_credits(figure)
    tight_plot(figure)
    #plt.show()

    return figure


def plot_fourier_convergence(time, signals, target_frequency, transient_time, unit):
    """ 
    *Plot fourier transforms over increasing time support*

    :param time 
    :type time: array
    :param signals
    :type signals: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param unit
    :type unit: string

    :returns:
        - **figure** - The plot of the temporal signal """

    # Could be done with a decorator ?
    if target_frequency == 0.0:
        target_frequency = set_default_target_freq(time)

    # Remove transient if needed, and truncate time signals to the correct
    # shape
    time, signals = clean_signals(time,
                                  signals,
                                  target_frequency,
                                  transient_time)

    # Total time should include an additional timestep
    total_time = time[-1] + time[1] - 2 * time[0]
    nb_periods = int(round(total_time * target_frequency, ROUND_PRECISION))
    nb_samples = time.size

    print("--- FOURIER CONVERGENCE ---")
    print(" ")
    print(" !!! Works on the average signal !!!")
    print(" ")
    print("--> Truncated time array")
    print(("    New initial time: {0:05.4f} s".format(time[0])))
    print(("    Number of periods: {0}".format(nb_periods)))
    print(("    Timestep: {0:05.4f} ms".format((time[1] - time[0]) * 1.e3)))

    average_signal = np.average(signals, axis=0)

    # Print complete spectra for complete 1/2 and 1/4 signal.
    freqs_complete, fft_complete, index_complete = get_clean_fft(
        time, average_signal, target_frequency)
    print(" ")
    print("--> Computed FFT over complete signal")
    print(("    Number of periods: {0}".format(nb_periods)))

    half_time, half_signal = clean_signals(time[:nb_samples // 2 + 1],
                                           average_signal[:nb_samples // 2 + 1],
                                           target_frequency,
                                           transient_time,
                                           trim_end=True)

    freqs_half, fft_half, index_half = get_clean_fft(
        half_time, half_signal, target_frequency)
    print(" ")
    print("--> Computed FFT over first half of signal")
    # print "    Initial time: {0:05.4f} s".format(half_time[0])
    half_periods = int(round(target_frequency / (freqs_half[1] - freqs_half[0]), ROUND_PRECISION))
    print(("    Number of periods: {0}".format(half_periods)))

    quarter_time, quarter_signal = clean_signals(time[:nb_samples // 4 + 1],
                                                 average_signal[:nb_samples // 4 + 1],
                                                 target_frequency,
                                                 transient_time,
                                                 trim_end=True)

    freqs_quarter, fft_quarter, index_quarter = get_clean_fft(
        quarter_time, quarter_signal, target_frequency)
    print(" ")
    print("--> Computed FFT over first quarter of signal")
    # print "    Initial time: {0:05.4f} s".format(quarter_time[0])
    quarter_periods = int(round(
        target_frequency / (freqs_quarter[1] - freqs_quarter[0]), ROUND_PRECISION))
    print(("    Number of periods: {0}".format(quarter_periods)))

    # Compute Fourier coefficient at target frequnecy over increasing
    # time_support
    cumul_fourier, cumul_periods = compute_cumulative_fourier(
        time,
        average_signal,
        target_frequency,
        transient_time)

    # Plot
    figure, axes = plt.subplots(2, 2, figsize=(8, 7))

    axes[0, 0].step(
        freqs_quarter,
        np.abs(fft_quarter),
        where='mid',
        color='orange',
        label='first quarter')
    axes[0, 0].step(
        freqs_half,
        np.abs(fft_half),
        where='mid',
        color='orangered',
        label='first half')
    axes[0, 0].step(
        freqs_complete,
        np.abs(fft_complete),
        where='mid',
        color='darkred',
        label='complete')

    axes[0, 0].axvline(
        x=freqs_complete[index_complete],
        linestyle='--',
        color='black')
    axes[0, 0].set_xlim(0, 3 * target_frequency)
    ymax = max(np.max(np.abs(fft_complete[index_complete - 1:index_complete + 2])),
               np.max(np.abs(fft_half[index_half - 1:index_half + 2])),
               np.max(np.abs(fft_quarter[index_quarter - 1:index_quarter + 2])))
    axes[0, 0].set_ylim(0.0, ymax * 1.2)
    axes[0, 0].set_xlabel('Frequency [Hz]')
    axes[0, 0].set_ylabel('Amplitude [{0}]'.format(unit))
    axes[0, 0].legend(loc=0, fontsize=8)
    axes[0, 0].set_title('Fourier spectrum - Standard scale')

    axes[0, 1].step(
        freqs_quarter,
        np.abs(fft_quarter),
        where='mid',
        color='orange',
        label='first quarter')
    axes[0, 1].step(
        freqs_half,
        np.abs(fft_half),
        where='mid',
        color='orangered',
        label='first half')
    axes[0, 1].step(
        freqs_complete,
        np.abs(fft_complete),
        where='mid',
        color='darkred',
        label='complete')

    axes[0, 1].axvline(
        x=freqs_complete[index_complete],
        linestyle='--',
        color='black')
    axes[0, 1].set_xlabel('Frequency [Hz]')
    axes[0, 1].set_ylabel('Amplitude [{0}]'.format(unit))
    axes[0, 1].legend(loc=0, fontsize=8)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_title('Fourier spectrum - Log scale')

    axes[1, 0].step(cumul_periods, np.abs(cumul_fourier), where='post')
    # TODO : add nominal value and +/- 10pct lines.
    # TODO add final value on plot
    axes[1, 0].set_xlabel('Number of periods [-]')
    axes[1, 0].set_ylabel('Amplitude [{0}]'.format(unit))
    axes[1, 0].set_title(_wrap_title(
        ('Fourier coefficient over increasing time support at '
         '{0:05.02f} Hz').format(target_frequency)))
    axes[1, 0].set_xticks(cumul_periods)
    axes[1, 0].set_ylim(0, np.max(np.abs(cumul_fourier)) * 1.08)
    axes[1, 0].grid(which='major', axis='x')
    axes[1, 0].text(cumul_periods[-1], np.abs(cumul_fourier[-1]),
                    'Converged value: {0:04.2f}'.format(np.abs(cumul_fourier[-1])),
                    horizontalalignment="right", verticalalignment="bottom")

    axes[1, 1].step(cumul_periods, _my_angle(cumul_fourier), where='post')
    # TODO : add nominal value and +/- 10pct lines.
    # TODO add final value on plot
    axes[1, 1].set_xlabel('Number of periods [-]')
    axes[1, 1].set_ylabel('Phase [rad]')
    axes[1, 1].set_title(_wrap_title(
        ('Fourier coefficient over increasing time support at '
         '{0:05.02f} Hz').format(target_frequency)))
    axes[1, 1].set_xticks(cumul_periods)
    axes[1, 1].set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    axes[1, 1].set_ylim(-0.08, 2 * np.pi * 1.08)
    axes[1, 1].grid(which='major', axis='both')
    axes[1, 1].text(cumul_periods[-1], _my_angle(cumul_fourier[-1]),
                    'Converged value: {0:04.2f}'.format(_my_angle(cumul_fourier[-1])),
                    horizontalalignment="right", verticalalignment="bottom")

    add_credits(figure)
    tight_plot(figure)

    return figure


def plot_psd_convergence(time, signals, target_frequency,
                         transient_time, discardphase, dbspl, unit,
                         nb_blocks, window_shape, overlap):
    """ 
    *Plot the convergence of the PSD*
    :param time 
    :type time: array
    :param signals
    :type signals: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param discardphase
    :type discardphase: bool
    :param dbspl
    :type dbspl: bool
    :param unit
    :type unit: string
    :param nb_blocks
    :type nb_blocks: int
    :param window_shape
    :type window_shape: string
    :param overlap
    :type overlap: str

    :returns:
        - **figure** - The plot of the temporal signal
        
    TO DO : gather this routine with plot_fourier_convergence ? """

    # Could be done with a decorator ?
    if target_frequency == 0.0:
        target_frequency = set_default_target_freq(time)

    # Remove transient if needed, and truncate time signals to the correct
    # shape
    time, signals = clean_signals(time,
                                  signals,
                                  target_frequency,
                                  transient_time)

    nb_periods = (time[-1] - time[0]) * target_frequency
    nb_samples = time.size

    print("--- PSD CONVERGENCE ---")
    print(" ")
    print(" !!! Works on the average signal !!!")
    print(" ")
    print("--> Truncated time array and uniformized time steps")
    print(("    New initial time: {0:05.4f} s".format(time[0])))
    print(("    Number of periods: {0}".format(nb_periods)))
    print(("    Timestep: {0:05.4f} ms".format((time[1] - time[0]) * 1.e3)))

    if discardphase:
        print(" ")
        print("--> Discard phase option activated !")

    psd_complete, freqs_complete, index_complete, energy_time, energy_freq = (
        get_average_psd(
            time,
            signals,
            target_frequency,
            transient_time,
            discardphase,
            nb_blocks,
            window_shape,
            overlap
        )
    )

    all_periods = int(round(target_frequency / (freqs_complete[1] - freqs_complete[0]), ROUND_PRECISION))
    print(" ")
    print("--> Computed PSD over complete signal")
    print(("    Number of periods: {0}".format(all_periods)))
    print(("    Average temporal power: {0}".format(energy_time)))
    print(("    Average spectral power: {0}".format(energy_freq)))

    half_time, half_signals = clean_signals(time[:nb_samples // 2],
                                            signals[:, :nb_samples // 2],
                                            target_frequency,
                                            transient_time,
                                            trim_end=True)

    psd_half, freqs_half, index_half, energy_time, energy_freq = (
        get_average_psd(
            half_time,
            half_signals,
            target_frequency,
            transient_time,
            discardphase,
            nb_blocks,
            window_shape,
            overlap
        )
    )

    print(" ")
    print("--> Computed PSD over first half of signal")
    # print "    Initial time: {0:05.4f} s".format(half_time[0])
    half_periods = int(round(target_frequency / (freqs_half[1] - freqs_half[0]), ROUND_PRECISION))
    print(("    Number of periods: {0}".format(half_periods)))
    print(("    Average temporal power: {0}".format(energy_time)))
    print(("    Average spectral power: {0}".format(energy_freq)))

    quarter_time, quarter_signals = clean_signals(time[:nb_samples // 4],
                                                  signals[:, :nb_samples // 4],
                                                  target_frequency,
                                                  transient_time,
                                                  trim_end=True)

    psd_quarter, freqs_quarter, index_quarter, energy_time, energy_freq = (
        get_average_psd(
            quarter_time,
            quarter_signals,
            target_frequency,
            transient_time,
            discardphase,
            nb_blocks,
            window_shape,
            overlap
        )
    )

    print(" ")
    print("--> Computed PSD over first quarter of signal")
    # print "    Initial time: {0:05.4f} s".format(quarter_time[0])
    quarter_periods = int(round(
        target_frequency / (freqs_quarter[1] - freqs_quarter[0]), ROUND_PRECISION))
    print(("    Number of periods: {0}".format(quarter_periods)))
    print(("    Average temporal power: {0}".format(energy_time)))
    print(("    Average spectral power: {0}".format(energy_freq)))

    # Plot
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))

    if dbspl:
        pref = (2.0e-5) ** 2
        psd_complete = 10. * np.log10(0.5 * psd_complete / pref)
        psd_half = 10. * np.log10(0.5 * psd_half / pref)
        psd_quarter = 10. * np.log10(0.5 * psd_quarter / pref)
        axes[1].set_ylabel('Spectral power [dB SPL / Hz]')
        axes[0].set_ylabel('Spectral power [db SPL / Hz]')
        axes[0].text(0.02,
                     0.02,
                     'WARNING: db SPL valid only for pressure !',
                     transform=axes[0].transAxes,
                     fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='red'))
        axes[1].text(0.02,
                     0.02,
                     'WARNING: db SPL valid only for pressure !',
                     transform=axes[1].transAxes,
                     fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='red'))
    else:
        axes[1].set_yscale('log')
        axes[1].set_ylabel(r'Spectral power [({0})$^2$/Hz]'.format(unit))
        axes[0].set_ylabel(r'Spectral power [({0})$^2$/Hz]'.format(unit))

    axes[0].step(
        freqs_quarter,
        np.abs(psd_quarter),
        where='mid',
        color='orange',
        label='first quarter')
    axes[0].step(
        freqs_half,
        np.abs(psd_half),
        where='mid',
        color='orangered',
        label='first half')
    axes[0].step(
        freqs_complete,
        np.abs(psd_complete),
        where='mid',
        color='darkred',
        label='complete')

    axes[0].axvline(x=target_frequency, linestyle='--', color='black')
    axes[0].set_xlim(0, 3 * target_frequency)
    ymax = max(np.abs(psd_complete[index_complete]),
               np.abs(psd_half[index_half]),
               np.abs(psd_quarter[index_quarter]))
    axes[0].set_ylim(0.0, ymax * 1.2)
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].legend(loc=0, fontsize=8)
    axes[0].set_title('PSD - Standard scale')

    axes[1].step(
        freqs_quarter,
        np.abs(psd_quarter),
        where='mid',
        color='orange',
        label='first quarter')
    axes[1].step(
        freqs_half,
        np.abs(psd_half),
        where='mid',
        color='orangered',
        label='first half')
    axes[1].step(
        freqs_complete,
        np.abs(psd_complete),
        where='mid',
        color='darkred',
        label='complete')

    axes[1].axvline(x=target_frequency, linestyle='--', color='black')
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].legend(loc=0, fontsize=8)
    axes[1].set_title('PSD - Log scale')
    axes[1].set_xscale('log')

    add_credits(figure)
    tight_plot(figure)

    return figure


def plot_fourier_variability(time, signals, target_frequency, transient_time, unit):
    """ 
    *Plot Fourier coefficients of the different signals*
    :param time 
    :type time: array
    :param signals
    :type signals: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param unit
    :type unit: string

    :returns:
        - **figure** - The plot of the Fourier variability analysis"""

    # Could be done with a decorator ?
    if target_frequency == 0.0:
        target_frequency = set_default_target_freq(time)

    # Remove transient if needed, and truncate time signals to the correct
    # shape
    time, signals = clean_signals(time,
                                  signals,
                                  target_frequency,
                                  transient_time)

    nb_periods = (time[-1] - time[0]) * target_frequency

    print("--- FOURIER VARIABILITY ---")
    print(" ")
    print("--> Truncated time array and uniformized time steps")
    print(("    New initial time: {0:05.4f} s".format(time[0])))
    print(("    Number of periods: {0}".format(nb_periods)))
    print(("    Timestep: {0:05.4f} ms".format((time[1] - time[0]) * 1.e3)))

    fourier_coeffs = get_coeff_fourier(
        time, signals, target_frequency, transient_time)

    print(" ")
    print(("--> Computed Fourier Coefficients at {0:03.03f} Hz".format(target_frequency)))
    print("")

    figure, axes = plt.subplots(2, 2, figsize=(8, 8))

    nb_signals = signals.shape[0]
    cmap = plt.cm.get_cmap("hsv", nb_signals + 1)  # For coloring probes

    mean_coeff = np.mean(fourier_coeffs)
    axes[1, 0].axhline(y=np.abs(mean_coeff), color='red', linewidth=2)
    for k, coeff in enumerate(fourier_coeffs):
        axes[1, 0].plot(
            k,
            np.abs(coeff),
            marker='o',
            markersize=8,
            color=cmap(k),
            linewidth=0,
            label='Probe {0}'.format(k))
    axes[1, 0].set_xlabel('Probe number')
    axes[1, 0].set_ylabel('Amplitude [{0}]'.format(unit))
    axes[1, 0].set_xlim(-1, nb_signals)
    axes[1, 0].set_xticks(list(range(0, nb_signals)))
    axes[1, 0].grid(axis='x')
    axes[1, 0].set_title(
        _wrap_title(
            'Amplitude of Fourier coefficients at {0:05.01f} Hz'.format(target_frequency)))

    axes[1, 1].axhline(y=_my_angle(mean_coeff), color='green', linewidth=2)
    for k, coeff in enumerate(fourier_coeffs):
        axes[1, 1].plot(
            k,
            _my_angle(coeff),
            marker='o',
            markersize=8,
            color=cmap(k),
            linewidth=0,
            label='Probe {0}'.format(k))
    axes[1, 1].set_xlim(-1, nb_signals)
    axes[1, 1].set_xlabel('Probe number')
    axes[1, 1].set_ylabel('Phase [rad]')
    axes[1, 1].set_xticks(list(range(0, nb_signals)))
    axes[1, 1].set_yticks([0, np.pi * 0.5, np.pi, 1.5 * np.pi, 2. * np.pi])
    axes[1, 1].grid(axis='both')
    axes[1, 1].set_title(
        _wrap_title(
            'Phase of Fourier coefficients at {0:05.01f} Hz'.format(target_frequency)))

    # Plot fluctuations vs averages
    time_averages = np.average(signals, axis=-1)
    mean_mean = np.average(time_averages)
    average_fourier = np.mean(fourier_coeffs)
    axes[0, 0].axhline(y=mean_mean, color='blue', linewidth=2)
    for k, (meanval, fourier_coeff) in enumerate(
            zip(time_averages, fourier_coeffs)):
        axes[0, 0].errorbar(k,
                            meanval,
                            yerr=np.abs(fourier_coeff),
                            fmt='o',
                            markersize=6,
                            color=cmap(k),
                            linewidth=1,
                            capsize=4,
                            label='Probe {0}'.format(k))
    xlim = axes[0, 0].get_xlim()
    rectangle = mpl.patches.Rectangle(xy=(xlim[0], mean_mean - np.abs(average_fourier)),
                                      width=xlim[1] - xlim[0],
                                      height=2 *
                                             np.abs(average_fourier),
                                      color='lightblue',
                                      alpha=0.5)
    axes[0, 0].add_patch(rectangle)
    axes[0, 0].set_xlabel('Probe number')
    axes[0, 0].set_ylabel('Amplitude [{0}]'.format(unit))
    axes[
        0,
        0].set_title(
        _wrap_title(
            'Average and fluctuations at {0:01.01f} Hz'.format(target_frequency)))
    axes[0, 0].set_xticks(list(range(k + 1)))
    xlim = axes[0, 0].get_xlim()
    ylim = axes[0, 0].get_ylim()
    axes[0, 0].text(xlim[0] + 1,
                    ylim[0] + 1,
                    _wrap_title('Horizontal line correspond to the average '
                                'value over time and probes. The average amp'
                                'litude of fluctuations is delimited by the '
                                'light blue zone.', 60),
                    fontsize=6,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='white'))
    axes[0, 0].grid(which='major', axis='x')
    if np.min(time_averages) > 0.0:
        axes[0, 0].set_ylim(0.0, None)

    # Circle of average velocity (over time and probes)
    circle = plt.Circle((0, 0), np.abs(mean_coeff),
                        fill=False, linewidth=2, edgecolor='red')
    axes[0, 1].add_artist(circle)

    # Draw 0, 45, 90 degrees lines for indication
    axes[0, 1].plot([0, 0], [-1.5 * np.abs(mean_coeff),
                             1.5 * np.abs(mean_coeff)], 'k--')
    axes[0, 1].plot([-1.5 * np.abs(mean_coeff), 1.5 *
                     np.abs(mean_coeff)], [0, 0], 'k--')
    axes[0, 1].plot([-1.5 * np.abs(mean_coeff), 1.5 * np.abs(mean_coeff)],
                    [-1.5 * np.abs(mean_coeff), 1.5 * np.abs(mean_coeff)], 'k:')
    axes[0, 1].plot([-1.5 * np.abs(mean_coeff), 1.5 * np.abs(mean_coeff)],
                    [1.5 * np.abs(mean_coeff), -1.5 * np.abs(mean_coeff)], 'k:')

    # Plot the fourier coefficient of the average of all velocity probes
    plot_point_x = np.real(mean_coeff)
    plot_point_y = np.imag(mean_coeff)
    axes[0, 1].plot([0, plot_point_x], [0, plot_point_y],
                    color='green', linewidth=2)
    for k, coeff in enumerate(fourier_coeffs):
        axes[0, 1].plot(
            np.real(coeff),
            np.imag(coeff),
            marker='o',
            markersize=6,
            color=cmap(k))
    # axes[0,1].set_xlim(
    #    (-1.5 * np.abs(mean_coeff),
    #     1.5 * np.abs(mean_coeff)))
    # axes[0,1].set_ylim(
    #    (-1.5 * np.abs(mean_coeff),
    #     1.5 * np.abs(mean_coeff)))
    axes[0, 1].set_xlabel("Real part")
    axes[0, 1].set_ylabel("Imaginary part")
    axes[0, 1].set_title(
        _wrap_title(
            'Fourier coefficients at {0:05.01f} Hz in complex space'.format(target_frequency)))

    add_credits(figure)
    tight_plot(figure)

    return figure


def plot_psd_variability(time, signals, target_frequency, transient_time,
                         nb_blocks=1, window_shape="flat", overlap=0.0):
    """ 
    *Plot PSD of each signal*

    :param time 
    :type time: array
    :param signals
    :type signals: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param nb_blocks
    :type nb_blocks: int
    :param window_shape
    :type window_shape: string
    :param overlap
    :type overlap: float
    

    :returns:
        - **figure** - The plot of the temporal signal"""

    # Could be done with a decorator ?
    if target_frequency == 0.0:
        target_frequency = set_default_target_freq(time)

    # Remove transient if needed, and truncate time signals to the correct
    # shape
    time, signals = clean_signals(time,
                                  signals,
                                  target_frequency,
                                  transient_time)

    nb_periods = (time[-1] - time[0]) * target_frequency
    nb_signals = signals.shape[0]

    print("--- PSD VARIABILITY ---")
    print(" ")
    print("--> Truncated time array and uniformized time steps")
    print(("    New initial time: {0:05.4f} s".format(time[0])))
    print(("    Number of periods: {0}".format(nb_periods)))
    print(("    Timestep: {0:05.4f} ms".format((time[1] - time[0]) * 1.e3)))

    figure, axes = plt.subplots(1, 1)

    psd_contributions = get_psd_harmonics(
        time, signals, target_frequency, transient_time,
        nb_blocks, window_shape, overlap
    )
    bottom = np.zeros(nb_signals)
    axes.bar(
        list(range(nb_signals)),
        psd_contributions[
        :,
        1],
        bottom=bottom,
        color='darkred',
        edgecolor='black',
        label='Target frequency')
    bottom += psd_contributions[:, 1]
    axes.bar(
        list(range(nb_signals)),
        psd_contributions[
        :,
        2],
        bottom=bottom,
        color='red',
        edgecolor='black',
        label='First harmonic')
    bottom += psd_contributions[:, 2]
    axes.bar(
        list(range(nb_signals)),
        psd_contributions[
        :,
        3],
        bottom=bottom,
        color='orange',
        edgecolor='black',
        label='Second harmonic')
    bottom += psd_contributions[:, 3]
    axes.bar(
        list(range(nb_signals)),
        psd_contributions[
        :,
        0],
        bottom=bottom,
        color='lightyellow',
        edgecolor='black',
        label='Other')
    axes.legend(loc=0, fontsize=8)

    axes.set_xticks(list(range(nb_signals)))
    axes.set_xlim(-0.5, nb_signals - 0.5)
    axes.set_title('PSD distributions of fluctuations')
    axes.set_xlabel('Probe number')
    axes.set_ylabel('Normalized power')

    add_credits(figure)
    tight_plot(figure)

    return figure


def write_output(time, signals, target_frequency,
                 transient_time, discardphase, unit, dbspl):
    """
    *Write output text file containing Fourier Transform and PSD of the average signal.*


    :param time 
    :type time: array
    :param signals
    :type signals: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param discardphase
    :type discardphase: bool
    :param unit
    :type unit: string
    :param dbspl
    :type dbspl: bool

    """

    # Could be done with a decorator ?
    if target_frequency == 0.0:
        target_frequency = set_default_target_freq(time)

    # Remove transient if needed, and truncate time signals to the correct
    # shape
    time, signals = clean_signals(time,
                                  signals,
                                  target_frequency,
                                  transient_time)
    average = np.average(signals, axis=0)  # average over all signals

    fft_freqs, fft_spectrum, _ = get_clean_fft(
        time, average, target_frequency, transient_time)

    psd_spectrum, _, _, _, _ = get_average_psd(
        time, signals, target_frequency, transient_time, discardphase)

    if dbspl: 
        pref = (2.0e-5) ** 2
        psd_spectrum = 10. * np.log10(0.5 * psd_spectrum / pref)

    data = np.array([fft_freqs,
                     np.abs(fft_spectrum),
                     np.angle(fft_spectrum),
                     psd_spectrum]).transpose()

    titles = ['Frequencies [Hz]',
              'DFT amplitudes [{0}]'.format(unit),
              'DFT phases [rad]',
              'PSD amplitudes [{0}^2/Hz]'.format(unit)]
    if dbspl:
        titles[-1] = "PSD amplitudes [db SPL / Hz]"

    titles = '#' + '    '.join(titles)

    comment = '# Phases included in the PSD of the average. \n'
    if discardphase:
        comment = '# Phase have been discarded when computing the average PSD \n'

    np.savetxt('fourier_psd.dat', data, header=titles, comments=comment)


if __name__ == '__main__':

    display_header()
    ARGS = read_param(sys.argv)
    TIME, SIGNALS = read_signals(ARGS.filename, ARGS.remove_average)

    if ARGS.subset:
        SIGNALS = extract_subset(ARGS.subset,SIGNALS)

    write_output(
        TIME,
        SIGNALS,
        ARGS.target_frequency,
        ARGS.transient_time,
        ARGS.discardphase,
        ARGS.unit,
        ARGS.dbspl)

    if ARGS.diagnostic == 'time':
        plot_time_signals(
            TIME,
            SIGNALS,
            ARGS.target_frequency,
            ARGS.transient_time,
            ARGS.unit)
    elif ARGS.diagnostic == 'fourier_convergence':
        plot_fourier_convergence(
            TIME,
            SIGNALS,
            ARGS.target_frequency,
            ARGS.transient_time,
            ARGS.unit)
    elif ARGS.diagnostic == 'psd_convergence':
        plot_psd_convergence(
            TIME,
            SIGNALS,
            ARGS.target_frequency,
            ARGS.transient_time,
            ARGS.discardphase,
            ARGS.dbspl,
            ARGS.unit,
            ARGS.nb_blocks,
            ARGS.window_shape,
            ARGS.overlap
        )
    elif ARGS.diagnostic == 'fourier_variability':
        plot_fourier_variability(
            TIME,
            SIGNALS,
            ARGS.target_frequency,
            ARGS.transient_time,
            ARGS.unit)
    elif ARGS.diagnostic == 'psd_variability':
        plot_psd_variability(
            TIME,
            SIGNALS,
            ARGS.target_frequency,
            ARGS.transient_time,
            ARGS.nb_blocks,
            ARGS.window_shape,
            ARGS.overlap
        )
