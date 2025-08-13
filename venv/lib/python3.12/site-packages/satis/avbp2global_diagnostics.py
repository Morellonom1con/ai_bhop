import sys
from textwrap import wrap

import numpy as np
import scipy.signal
import scipy.stats

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec
    import matplotlib.patches
    from matplotlib.ticker import FormatStrFormatter
except ImportError:
    print("Matplotlib package is not available !")
    sys.exit(-1)

from satis.satis_library import (remove_average, get_clean_fft, define_good_time_array, get_clean_average, get_coeff_fourier,
                       get_cumulative_average, compute_cumulative_fourier, get_psd_harmonics,
                       apply_butter_bandpass_filter, interpolate_signals)
from satis import avbp2global_compute_ftf

__all__ = ["step1_diagnostic", "step2_diagnostic", "step3_diagnostic", "step4_diagnostic"]


# For plots

# def _add_periodic_grid(pyplotax, target_frequency)

def _wrap_title(title, length=40):
    ''' Wrap text '''
    return "\n".join(wrap(title, length))


# TODO : find better name ?
def _my_angle(complex_array):
    ''' Return the angles of arrays, between 0 and 2pi '''
    angles = np.mod(np.angle(complex_array), 2 * np.pi)
    return angles


def _get_fft_fluctuations(
        time, heat_release, velocities, parameters):
    ''' Compute the Fast Fourier Transform of the fluctuations of heat release
         and velocities at probes. '''

    # Extract useful parameters
    target_frequency = parameters['target_frequency']
    transient_time = parameters['transient_time']

    # Extract fluctuations
    heat_release = remove_average(heat_release)
    velocities = remove_average(velocities)

    # Obtain the FFT of heat release and velocities
    fft_frequencies, fft_heat_release, _ = get_clean_fft(
        time, heat_release, target_frequency, transient_time)
    _, fft_velocities, _ = get_clean_fft(
        time, velocities, target_frequency, transient_time)

    return fft_frequencies, fft_heat_release, fft_velocities


def step1_diagnostic(time, heat_release, velocities, parameters):
    ''' Perform step 1 diagnostics, namely:
        - A plot of the heat release signal in time
        - A plot of the velocity signals in time
        - A plot of the fourier frequency spectra for heat release and average velocity

        The inputs are:
        - heat_release : contains time and signal of heat release.
        - velocities : contains the pairs (time, velocity) for each probe.
        - parameters : input arguments '''

    # Print some information
    print(" ")
    print("--- STEP 1 ---")
    print("")

    # Get truncated time array
    transient_time = parameters['transient_time']
    target_frequency = parameters['target_frequency']
    clean_time = define_good_time_array(
        time, target_frequency, transient_time)
    nb_periods = (clean_time[-1] - clean_time[0]) * target_frequency
    print("--> Removed transient regime and uniformized time steps:")
    print("    Initial time:      {0:05.05f} s".format(clean_time[0]))
    print("    End time:          {0:05.05f} s".format(clean_time[-1]))
    print("    Time step:         {0:05.05f} ms".format(
        (clean_time[1] - clean_time[0]) * 1e3))
    print("    Number of periods: {0}".format(nb_periods))
    print("")

    # Compute mean velocity
    mean_velocity = np.average(velocities, axis=0)

    # Compute clean time-averages
    mean_heat_release = get_clean_average(
        time, heat_release, target_frequency, transient_time)
    mean_mean_velocity = get_clean_average(
        time, mean_velocity, target_frequency, transient_time)

    # Compute fourier transforms
    fft_frequencies, fft_heat_release, fft_mean_velocity = (
        _get_fft_fluctuations(
            time, heat_release, mean_velocity, parameters))

    # Print some information
    nyquist_frequency = fft_frequencies[-1]
    frequency_resolution = fft_frequencies[1] - fft_frequencies[0]

    print("--> Computed the Fourier Transforms")
    print("    Nyquist frequency: {0:10.5f}".format(nyquist_frequency))
    print("    Frequency resolution: {0:10.5f}".format(frequency_resolution))
    print("")

    # Create matplotlib figure
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot heat release fluctuations with time
    heat_release_fluct = remove_average(heat_release)
    axes[0].plot(time, heat_release_fluct, color='black')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude [W]')
    axes[0].set_title(r'Global heat release fluctuations $Q(t)$')
    nb_periods = int((clean_time[-1] - clean_time[0]) * target_frequency)
    axes[0].set_xticks(np.linspace(
        clean_time[0], clean_time[-1], nb_periods + 1))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%01.03f'))
    axes[0].grid(which='major', axis='x')
    xlim = axes[0].get_xlim()
    ylim = axes[0].get_ylim()
    rectangle = matplotlib.patches.Rectangle(xy=(xlim[0], ylim[0]),
                                             width=clean_time[0] - xlim[0],
                                             height=ylim[1] - ylim[0],
                                             color='grey',
                                             hatch='/',
                                             alpha=0.5)
    axes[0].add_patch(rectangle)

    # Plot velocity fluctuations with time
    velocities_fluct = remove_average(velocities)
    cmap = plt.cm.get_cmap("hsv", velocities.shape[0] + 1)
    for k, velocity_fluct in enumerate(velocities_fluct):
        axes[1].plot(time, velocity_fluct, color=cmap(k))
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Amplitude [m/s]')
    axes[1].set_title(r'Velocity fluctuations at probes $u_i(t)$')
    nb_periods = int((clean_time[-1] - clean_time[0]) * target_frequency)
    axes[1].set_xticks(np.linspace(
        clean_time[0], clean_time[-1], nb_periods + 1))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%01.03f'))
    axes[1].grid(which='major', axis='x')
    xlim = axes[1].get_xlim()
    ylim = axes[1].get_ylim()
    rectangle = matplotlib.patches.Rectangle(xy=(xlim[0], ylim[0]),
                                             width=clean_time[0] - xlim[0],
                                             height=ylim[1] - ylim[0],
                                             color='grey',
                                             hatch='/',
                                             alpha=0.5)
    axes[1].add_patch(rectangle)

    # Plot fourier spectra of heat release and mean velocity
    target_frequency = parameters['target_frequency']
    axes[2].set_xlabel('Frequency [Hz]')
    axes[2].set_ylabel('Amplitude [-]')
    axes[2].set_title("\n".join(wrap('Fourier coefficients '
                                     'normalized by time-average', 40)))
    axes[2].set_xlim([0, 3 * target_frequency])
    fft_heat_release = np.abs(fft_heat_release) / \
                       mean_heat_release
    axes[2].step(
        fft_frequencies,
        fft_heat_release,
        color='black',
        label='Heat release',
        where='mid')
    fft_mean_velocity = np.abs(fft_mean_velocity) / \
                        mean_mean_velocity
    axes[2].step(
        fft_frequencies,
        fft_mean_velocity,
        color='red',
        label='Mean velocity',
        where='mid')
    axes[2].axvline(x=target_frequency, color='grey', linestyle='--')
    axes[2].legend(loc=0, fontsize=6)
    axes[2].text(x=0.9,
                 y=0.6,
                 s='Frequency resolution = {0:02.02f} Hz'.format(frequency_resolution),
                 transform=axes[2].transAxes,
                 horizontalalignment='right',
                 verticalalignment='top',
                 fontsize=6)
    axes[2].text(x=0.9,
                 y=0.5,
                 s='Number of periods = {0}'.format(nb_periods),
                 transform=axes[2].transAxes,
                 horizontalalignment='right',
                 verticalalignment='top',
                 fontsize=6)

    # Save figure if asked, before showing it
    debug = parameters['debug']
    if debug:
        figure.savefig('step1_diagnostic.eps')
        # Replace .eps by sth else to change format
    figure.tight_layout()

    print("--> Displaying matplotlib figure")
    print("")

    plt.show()


def step2_diagnostic(time, heat_release, velocities, parameters):
    ''' Plot the Fourier coefficients of all velocity probes to
         identify and discard bad probes. '''

    def _plot_probe_amplitudes(pyplotax, fourier_coeffs, cmap):
        ''' Plot all probe fourier amplitudes '''
        mean_coeff = np.mean(fourier_coeffs)
        pyplotax.axhline(y=np.abs(mean_coeff), color='red', linewidth=2)
        for k, coeff in enumerate(fourier_coeffs):
            pyplotax.plot(
                k,
                np.abs(coeff),
                marker='o',
                markersize=8,
                color=cmap(k),
                linewidth=0,
                label='Probe {0}'.format(k))
        pyplotax.set_xlabel('Probe number')
        pyplotax.set_ylabel('Amplitude [m/s]')
        ylimits = pyplotax.get_ylim()
        pyplotax.set_ylim(0.0, ylimits[1])
        pyplotax.set_xticks(list(range(k + 1)))
        pyplotax.grid(which='major', axis='x')

        return

    def _plot_probe_phases(pyplotax, fourier_coeffs, cmap):
        ''' Plot all probe fourier phases '''
        mean_coeff = np.mean(fourier_coeffs)
        pyplotax.axhline(y=_my_angle(mean_coeff), color='green', linewidth=2)
        for k, coeff in enumerate(fourier_coeffs):
            pyplotax.plot(
                k,
                _my_angle(coeff),
                marker='o',
                markersize=8,
                color=cmap(k),
                linewidth=0,
                label='Probe {0}'.format(k))
        pyplotax.set_xlabel('Probe number')
        pyplotax.set_ylabel('Phase [rad]')
        pyplotax.set_yticks([0., .5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
        pyplotax.set_xticks(list(range(k + 1)))
        pyplotax.grid(which='major', axis='both')

        return

    def _plot_complex_space(pyplotax, fourier_coeffs, fourier_coeff_ref, cmap):
        ''' Plot fourier phase times temporal average in complex space '''

        # Averages
        fourier_mean_coeff = np.mean(fourier_coeffs)

        # Circle of average velocity (over time and probes)
        circle = plt.Circle((0, 0), np.abs(fourier_mean_coeff),
                            fill=False, linewidth=2, edgecolor='purple')
        pyplotax.add_artist(circle)

        # Draw 0, 45, 90 degrees lines for indication
        pyplotax.plot([0, 0], [-1.5 * np.abs(fourier_mean_coeff),
                               1.5 * np.abs(fourier_mean_coeff)], 'k--')
        pyplotax.plot([-1.5 * np.abs(fourier_mean_coeff), 1.5 *
                       np.abs(fourier_mean_coeff)], [0, 0], 'k--')
        pyplotax.plot([-1.5 * np.abs(fourier_mean_coeff), 1.5 * np.abs(fourier_mean_coeff)],
                      [-1.5 * np.abs(fourier_mean_coeff), 1.5 * np.abs(fourier_mean_coeff)], 'k:')
        pyplotax.plot([-1.5 * np.abs(fourier_mean_coeff), 1.5 * np.abs(fourier_mean_coeff)],
                      [1.5 * np.abs(fourier_mean_coeff), -1.5 * np.abs(fourier_mean_coeff)], 'k:')

        # Plot the fourier coefficient of the average of all velocity probes
        plot_point_x = np.real(fourier_mean_coeff)
        plot_point_y = np.imag(fourier_mean_coeff)
        pyplotax.plot([0, plot_point_x], [0, plot_point_y],
                      color='green', linewidth=2)
        for k, coeff in enumerate(fourier_coeffs):
            pyplotax.plot(
                np.real(coeff),
                np.imag(coeff),
                marker='o',
                markersize=6,
                color=cmap(k))
        pyplotax.plot(
            np.real(fourier_coeff_ref),
            np.imag(fourier_coeff_ref),
            marker='s',
            markersize=6,
            color='black',
            linewidth=0,
            label='Heat release')
        pyplotax.set_xlim(
            (-1.5 * np.abs(fourier_mean_coeff),
             1.5 * np.abs(fourier_mean_coeff)))
        pyplotax.set_ylim(
            (-1.5 * np.abs(fourier_mean_coeff),
             1.5 * np.abs(fourier_mean_coeff)))
        pyplotax.set_xlabel("Real part")
        pyplotax.set_ylabel("Imaginary part")

        return

    # Print some information
    print(" ")
    print("--- STEP 2 ---")
    print("")
    print("--> Removed transient regime and uniformized time steps:")
    print("    Initial time: {0:05.05f} s".format(time[0]))
    print("    End time:     {0:05.05f} s".format(time[-1]))
    print("    Time step:    {0:05.05f} ms".format((time[1] - time[0]) * 1e3))
    print("")

    # Compute Fourier coeffs at target_frequency for all velocity probes
    target_frequency = parameters['target_frequency']
    transient_time = parameters['transient_time']
    fourier_velocities = get_coeff_fourier(
        time, velocities, target_frequency, transient_time)
    fourier_heat_release = get_coeff_fourier(
        time, heat_release, target_frequency, transient_time)

    print("--> Computed Fourier Coefficients at {0:03.03f} Hz".format(target_frequency))
    print("")

    # Get time averages with clean time array
    mean_velocities = get_clean_average(
        time, velocities, target_frequency, transient_time)
    mean_heat_release = get_clean_average(
        time, heat_release, target_frequency, transient_time)

    print("--> Computed temporal averages")
    print("")

    debug = parameters['debug']
    if debug:
        np.savetxt(('fourier_coefficients_velocity_probes_'
                    '{0:01.0f}Hz.dat').format(target_frequency),
                   np.array([np.abs(fourier_velocities),
                             _my_angle(fourier_velocities)]).transpose())

    figure, axes = plt.subplots(2, 2, figsize=(8, 8))
    nb_probes = fourier_velocities.shape[0]

    cmap = plt.cm.get_cmap("hsv", nb_probes + 1)  # For coloring probes

    # Plot all amplitudes
    _plot_probe_amplitudes(axes[1, 0], fourier_velocities, cmap)
    axes[1, 0].set_title(
        _wrap_title(
            ('Fourier coefficients of velocity fluctuations at '
             '{0:01.1f} Hz').format(target_frequency)))

    # Plot all phases
    _plot_probe_phases(axes[1, 1], fourier_velocities, cmap)
    axes[1, 1].set_title(
        _wrap_title(
            ('Fourier coefficients of velocity fluctuations at '
             '{0:01.1f} Hz').format(target_frequency)))

    # Plot velocity fluctuations versus velocity averages.
    mean_velocity = np.mean(mean_velocities)
    mean_fourier_velocity = np.mean(fourier_velocities)
    axes[0, 0].axhline(y=mean_velocity, color='blue', linewidth=2)
    for k, (meanval, fourier_coeff) in enumerate(
            zip(mean_velocities, fourier_velocities)):
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
    rectangle = matplotlib.patches.Rectangle(xy=(xlim[0], mean_velocity - np.abs(mean_fourier_velocity)),
                                             width=xlim[1] - xlim[0],
                                             height=2 *
                                                    np.abs(mean_fourier_velocity),
                                             color='lightblue',
                                             alpha=0.5)
    axes[0, 0].add_patch(rectangle)
    axes[0, 0].set_xlabel('Probe number')
    axes[0, 0].set_ylabel('Amplitude [m/s]')
    axes[
        0,
        0].set_title(
        _wrap_title(
            'Velocity average and fluctuations at {0:01.01f}'.format(target_frequency)))
    axes[0, 0].set_ylim(0.0, None)
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

    # Circle of average velocity (over time and probes)
    fourier_velocities = fourier_velocities / mean_velocity
    fourier_heat_release = fourier_heat_release / mean_heat_release
    _plot_complex_space(
        axes[
            0,
            1],
        fourier_velocities,
        fourier_heat_release,
        cmap)
    axes[0, 1].set_title(
        _wrap_title("Fourier coefficients in complex space "
                    "(normalized by average over time and probes)"))
    axes[0, 1].legend(loc=0, fontsize=6)

    print("--> Displaying matplotlib figure")
    print("")

    figure.tight_layout()
    plt.show()

    return


def step3_diagnostic(time, heat_release, velocities, parameters):
    ''' Compute and display diagnostics to check if
         enough time signal is used. '''

    # Get useful parameters
    target_frequency = parameters['target_frequency']
    transient_time = parameters['transient_time']
    debug = parameters['debug']

    print(" ")
    print("--- STEP 3 ---")
    print("")
    print("--> Removed transient regime and uniformized time steps:")
    print("    Initial time: {0:05.05f} s".format(time[0]))
    print("    End time:     {0:05.05f} s".format(time[-1]))
    print("    Time step:    {0:05.05f} ms".format((time[1] - time[0]) * 1e3))

    mean_velocity = np.average(velocities, axis=0)

    # Compute cumulative moving average of velocities
    cma_mean_velocity = get_cumulative_average(time, mean_velocity, target_frequency, transient_time)
    cma_heat_release = get_cumulative_average(time, heat_release, target_frequency, transient_time)
    cma_mean_velocity /= cma_mean_velocity[-1]
    cma_heat_release /= cma_heat_release[-1]

    print(" ")
    print("--> Computed cumulative averages.")

    # Compute moving fourier coefficients of the mean velocity
    moving_fourier_mean_velocity, periods = compute_cumulative_fourier(
        time, mean_velocity, target_frequency, transient_time)
    moving_fourier_heat_release, _ = compute_cumulative_fourier(
        time, heat_release, target_frequency, transient_time)

    nb_periods = (time[-1] - time[0]) * target_frequency
    print(" ")
    print("--> Computed Fourier Coefficient over increasing time support.")
    print("    Target frequency:  {0:03.00f}".format(target_frequency))
    print("    Number of periods: {0}".format(nb_periods))

    # Compute contributions from harmonics of target_frequency to the total
    # PSD power

    # TODO !!! Correct mistake ? Should take the average of PSDs and not the PSD od the average !
    # To check in spectral tool...
    harmonics_power_mean_velocity = get_psd_harmonics(
        time, mean_velocity, target_frequency, transient_time)
    harmonics_power_heat_release = get_psd_harmonics(
        time, heat_release, target_frequency, transient_time)

    print(" ")
    print("--> Computed Power Spectral Density and its distribution.")

    # If save option is activated, write output text files
    if debug:
        np.savetxt('cumulative_averages_velocities.dat',
                   np.vstack((time, cma_velocities)).transpose())
        np.savetxt('moving_fourier_heat_release.dat', np.vstack(
            (periods, moving_fourier_heat_release)).transpose())
        np.savetxt('moving_fourier_mean_velocity.dat', np.vstack(
            (periods, moving_fourier_mean_velocity)).transpose())
        np.savetxt(
            'harmonics_power_mean_velocity.dat',
            harmonics_power_mean_velocity)
        np.savetxt(
            'harmonics_power_heat_release.dat',
            harmonics_power_heat_release)

    # Plot all diagnostics, using gridspec to deal with non-grid subplots
    figure = plt.figure(figsize=(12, 6))
    # Control subplot positions
    plotgrid = matplotlib.gridspec.GridSpec(2, 3)

    # Plot the cumulative moving averages
    ax0 = figure.add_subplot(plotgrid[:, 0])
    ax0.step(periods,
             cma_mean_velocity,
             label='Mean velocity',
             color='red',
             where='post')
    ax0.step(periods,
             cma_heat_release,
             label='Heat release',
             color='black',
             where='post')
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('Amplitude [-]')
    nb_periods = int((time[-1] - time[0]) * target_frequency)
    ax0.set_xticks(list(range(nb_periods + 1)))
    ax0.grid(which='major', axis='x')
    ax0.legend(loc=0, fontsize=6)
    ax0.set_title(_wrap_title('Cumulative average normalized by global time-average'))

    # Plot the moving Fourier coefficients
    ax1 = figure.add_subplot(plotgrid[0, 1])
    # Use different y-scale for phase
    ax1b = figure.add_subplot(plotgrid[1, 1])
    # Normalize by module of last value to obtain comparable modules for
    # velocity and heat release
    moving_fourier_mean_velocity /= np.abs(np.mean(mean_velocity))
    moving_fourier_heat_release /= np.abs(np.mean(heat_release))
    ax1.step(
        periods,
        np.abs(moving_fourier_mean_velocity),
        color='red',
        linestyle='-',
        label='Mean velocity module',
        where='post')
    ax1b.step(
        periods,
        _my_angle(moving_fourier_mean_velocity),
        color='red',
        linestyle='--',
        label='Mean velocity phase',
        where='post')
    ax1.step(
        periods,
        np.abs(moving_fourier_heat_release),
        color='black',
        linestyle='-',
        label='Heat release module',
        where='post')
    ax1b.step(
        periods,
        _my_angle(moving_fourier_heat_release),
        color='black',
        linestyle='--',
        label='Heat release phase',
        where='post')
    ax1.set_xlabel('Number of periods used')
    ax1.set_ylabel('Module [-]')
    ax1b.set_xlabel('Number of periods used')
    ax1b.set_ylabel('Phase [rad]')
    ax1.set_title(
        _wrap_title(('Normalized Fourier coefficients with increasing '
                     'time support')))
    ax1.set_ylim(0.0, None)
    ax1b.set_yticks([0., .5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])

    ax1.set_xticks(list(range(nb_periods + 1)))
    ax1b.set_xticks(list(range(nb_periods + 1)))
    ax1.grid(which='major', axis='x')
    ax1b.grid(which='major', axis='x')

    ax1.legend(loc=0, fontsize=6)
    ax1b.legend(loc=0, fontsize=6)

    # Plot PSD contributions
    ax2 = figure.add_subplot(plotgrid[0, 2])
    ax2b = figure.add_subplot(plotgrid[1, 2])

    labels = 'Other', r'$f_0$', r'$2f_0$', r'$3f_0$'
    colors = ['white', 'gold', 'yellowgreen', 'lightskyblue']
    explode = (0, 0.1, 0, 0)  # explode 2nd slice corresponding to freq

    # Mean velocity
    ax2.pie(harmonics_power_mean_velocity,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=140)
    ax2.axis('equal')
    ax2.set_title(_wrap_title('Power distribution for mean velocity'))

    # Heat release
    ax2b.pie(harmonics_power_heat_release,
             explode=explode,
             labels=labels,
             colors=colors,
             autopct='%1.1f%%',
             shadow=True,
             startangle=140)
    ax2b.axis('equal')
    ax2b.set_title(_wrap_title('Power distribution for heat release'))

    figure.tight_layout()
    print(" ")
    print("--> Displaying matplotlib figure.")

    plt.show()

    return


def step4_diagnostic(time, heat_release, velocities, parameters):
    ''' Plot diagnostics to check the quality of the Flame Transfer Function.
        Diagnostic 1: Check the fourier spectrum of heat release and
        gain * velocity (should be equal at the target frequency).
        Diagnostic 2: Plot temporal signals of heat_release(t) vs
        gain * velocity (t-delay).
        Diagnostic 3: Phase diagram of heat_release as a function of velocity.
        Should follow a theoretical curve given by the Flame Transfer Function.

        For diagnostics 2 and 3, bandpass filtering can be applied around
        the frequency of interest.'''

    def _get_filtered_fluctuations(time, heat_release, velocities, parameters):
        ''' Apply bandpass filtering to heat release and velocity fluctuations. '''
        filter_bandwidth = parameters['filter_bandwidth']
        target_frequency = parameters['target_frequency']
        frequency_resolution = 1. / (time[-1] - time[0])
        nyquist_frequency = 0.5 / (time[1] - time[0])
        heat_release = remove_average(heat_release)
        velocities = remove_average(velocities)

        if filter_bandwidth > 0.0:
            low_frequency = max(0.0, target_frequency - 0.5 * filter_bandwidth)
            high_frequency = min(
                target_frequency +
                0.5 *
                filter_bandwidth,
                nyquist_frequency)
            filtered_velocities = apply_butter_bandpass_filter(
                time, velocities, low_frequency, high_frequency)
            filtered_heat_release = apply_butter_bandpass_filter(
                time, heat_release, low_frequency, high_frequency)
        else:
            low_frequency = 0.0
            high_frequency = nyquist_frequency
            filtered_velocities = velocities
            filtered_heat_release = heat_release

        frequency_limits = [low_frequency, high_frequency]

        return filtered_heat_release, filtered_velocities, frequency_limits

    print(" ")
    print("--- STEP 4 ---")
    print("")
    print("--> Removed transient regime and uniformized time steps:")
    print("    Initial time: {0:05.05f} s".format(time[0]))
    print("    End time:     {0:05.05f} s".format(time[-1]))
    print("    Time step:    {0:05.05f} ms".format((time[1] - time[0]) * 1e3))
    print("")

    mean_velocity = np.average(velocities, axis=0)

    # Get useful parameters
    target_frequency = parameters['target_frequency']
    transient_time = parameters['transient_time']

    mean_mean_velocity = get_clean_average(
        time, mean_velocity, target_frequency, transient_time)
    mean_heat_release = get_clean_average(
        time, heat_release, target_frequency, transient_time)

    # Get Fourier spectra of heat release and mean velocity fluctuations
    fft_frequencies, fft_heat_release, fft_mean_velocity = _get_fft_fluctuations(
        time, heat_release, mean_velocity, parameters)

    nyquist_frequency = fft_frequencies[-1]
    frequency_resolution = fft_frequencies[1] - fft_frequencies[0]

    print("--> Computed the Fourier Transforms")
    print("    Nyquist frequency: {0:10.5f}".format(nyquist_frequency))
    print("    Frequency resolution: {0:10.5f}".format(frequency_resolution))
    print("")

    # Get the Flame Transfer Function
    ftf_gain, ftf_delay, _ = avbp2global_compute_ftf.get_flame_transfer_function(
        time, heat_release, mean_velocity, parameters)

    print("--> Computed the FTF.")
    print("")

    # Filter heat release and mean velocity fluctuations if required
    filtered_heat_release, filtered_velocities, frequency_limits = _get_filtered_fluctuations(
        time, heat_release, velocities, parameters)
    filtered_mean_velocity = np.average(filtered_velocities, axis=0)
    low_frequency = frequency_limits[0]
    high_frequency = min(frequency_limits[1], 3 * target_frequency)

    print("--> Computed the filtered fluctuations.")
    print(("    Filter bandwidth:"
           " {0:05.04f} - {1:05.04f} Hz").format(*frequency_limits))
    print("")

    # Plot diagnostics
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    cmap = plt.cm.get_cmap("hsv", velocities.shape[0] + 1)

    # Plot the fourier spectra
    # TODO : normalize by average values
    plot_heat_release = np.abs(fft_heat_release) / \
                        mean_heat_release
    axes[0].step(
        fft_frequencies,
        plot_heat_release,
        color='red',
        label='Heat release',
        where='mid')
    plot_mean_velocity = \
        np.abs(fft_mean_velocity) / mean_mean_velocity
    axes[0].step(
        fft_frequencies,
        plot_mean_velocity,
        color='black',
        label='Mean velocity',
        where='mid')
    axes[0].set_xlim(0, 3 * target_frequency)
    # axes[0].set_ylim(0, 1.1)
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(_wrap_title('Normalized Fourier Amplitudes'))
    axes[0].legend(loc=0, fontsize=6)

    # Hatch the frequencies removed by the filter
    rectangle = matplotlib.patches.Rectangle(xy=(0., 0.),
                                             width=low_frequency,
                                             height=1.1,
                                             color='red',
                                             hatch='/',
                                             alpha=0.5)
    axes[0].add_patch(rectangle)
    rectangle = matplotlib.patches.Rectangle(xy=(high_frequency, 0.),
                                             width=3 * target_frequency - high_frequency,
                                             height=1.1,
                                             color='red',
                                             hatch='/',
                                             alpha=0.5)
    axes[0].add_patch(rectangle)

    # Plot the temporal heat release + temporal velocity with FTF gain and
    # delay, after filtering
    factor = 1. / np.max(np.abs(filtered_heat_release))
    for k, velocity in enumerate(filtered_velocities):
        axes[1].plot(
            time + ftf_delay,
            ftf_gain * velocity * factor,
            color=cmap(k),
            linestyle='-',
            label='Probe {0}'.format(k))
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Normalized fluctuations [-]')
    axes[1].set_title(
        _wrap_title('Heat release fluctuations vs FTF * velocity fluctuations'))
    axes[1].plot(
        time,
        filtered_heat_release *
        factor,
        color='black',
        linestyle='-',
        label='Heat release')
    nb_periods = int((time[-1] - time[0]) * target_frequency)
    axes[1].set_xticks(np.linspace(time[0], time[-1], nb_periods + 1))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%01.03f'))
    axes[1].grid(which='major', axis='x')

    # Plot phase diagram (heat_release vs mean_velocity)

    # Theoretical curve (obtained if signals are perfectly related by the FTF)
    theory_time = np.linspace(0., 1. / target_frequency, 100)
    theory_velocity = np.cos(-2. * np.pi * target_frequency * theory_time)
    theory_heat_release = np.cos(-2. * np.pi *
                                 target_frequency * (theory_time - ftf_delay))
    axes[2].plot(
        theory_velocity,
        theory_heat_release,
        color='deepskyblue',
        linestyle='--')

    # Compute Kernel Density Estimation (KDE)
    # TODO how is this done ? and is it correct ?
    velocity_envelop = np.abs(scipy.signal.hilbert(
        filtered_mean_velocity))  # velocity envelop
    heat_release_envelop = np.abs(
        scipy.signal.hilbert(filtered_heat_release))  # HR envelop
    # Normalized by the envelop in time
    filtered_mean_velocity = np.divide(
        filtered_mean_velocity, velocity_envelop)
    # Normalized by the envelop in time
    filtered_heat_release = np.divide(
        filtered_heat_release, heat_release_envelop)
    # To compare amplitudes of q(t) and n*u(t)
    factor = np.mean(velocity_envelop) / np.mean(heat_release_envelop)
    filtered_mean_velocity = ftf_gain * factor * filtered_mean_velocity
    umin = filtered_mean_velocity.min()  # why ?
    umax = filtered_mean_velocity.max()  # why ?
    qmin = filtered_heat_release.min()  # why ?
    qmax = filtered_heat_release.max()  # why ?
    UU, QQ = np.mgrid[umin:umax:100j, qmin:qmax:100j]
    # UU, QQ = np.mgrid[-1:1:100j, -1:1:100j]
    positions = np.vstack([UU.ravel(), QQ.ravel()])
    values = np.vstack([filtered_mean_velocity, filtered_heat_release])
    kernel = scipy.stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, UU.shape)
    Z = Z / np.amax(Z)

    # axes[2].imshow(np.rot90(Z), cmap='hot', extent=[umin, umax, qmin, qmax],
    axes[2].imshow(np.rot90(Z), cmap='hot', extent=[-1, 1, -1, 1],
                   clim=(0.0, 1.0))
    axes[2].set_xlabel(r'$N \times u(t)$')
    axes[2].set_ylabel(r'$Q(t)$')
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(-1, 1)
    axes[2].set_title('Quality polar plot')
    axes[2].text(1.0, 1.0, r'$N_2$={0:01.1f} J/m - $\tau$={1:01.3f} ms'.format(ftf_gain, ftf_delay * 1e3),
                 horizontalalignment='right',
                 verticalalignment='top',
                 color='white',
                 fontsize=8)

    figure.tight_layout()

    print("--> Displaying matplotlib figure.")
    plt.show()

    return


def diagnostic(raw_heat_release_data, raw_velocities_data, parameters):
    ''' Format raw heat release and velocity data correctly and choose
         diagnostic to perform (step 1, step 2, step 3 or step 4). '''

    # TODO Check dimensions of raw data

    time = raw_heat_release_data[0, :]
    heat_release = raw_heat_release_data[1, :]
    velocities = raw_velocities_data[:, 1, :]

    # Launch adequate step diagnostic
    step = parameters['step']
    if step != 1:
        target_frequency = parameters['target_frequency']
        transient_time = parameters['transient_time']
        new_time = define_good_time_array(
            time, target_frequency, transient_time)
        heat_release = interpolate_signals(time, heat_release, new_time)
        velocities = interpolate_signals(time, velocities, new_time)
        time = new_time

    if step == 1:
        step1_diagnostic(time, heat_release, velocities, parameters)
    elif step == 2:
        step2_diagnostic(time, heat_release, velocities, parameters)
    elif step == 3:
        step3_diagnostic(time, heat_release, velocities, parameters)
    elif step == 4:
        step4_diagnostic(time, heat_release, velocities, parameters)

    return
