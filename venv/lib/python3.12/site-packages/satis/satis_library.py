import sys
import numpy as np
import scipy.interpolate
import scipy.signal

ROUND_PRECISION = 10


def remove_initial_transient(time_array, new_initial_time):
    """ 
    *Remove a portion "transient_time" of time_array, corresponding to a
    transient regime*

    :param time_array
    :type time_array: array
    :param new_initial_time
    :type new_initial_time: float

    :returns:
        - **new_time_array** - array starting at new_initial_time


    """

    initial_time = time_array[0]
    end_time = time_array[-1]

    # Check that transient_time is consistently defined.
    if new_initial_time >= end_time:
        print((("     New initial time {0:05.4f} is larger than the end time "
               "{1:05.4f} the signal.").format(new_initial_time, end_time)))
        print("     Please check option --tinit. ")
        raise ValueError("Invalid new initial time specified with -t / --tinit")
    if new_initial_time <= initial_time:
        new_initial_time = initial_time

    # Define the new time array
    new_time_array = time_array[time_array >= new_initial_time]

    return new_time_array


def define_good_time_array(time_array, frequency, new_initial_time=0.0, trim_end=False):
    """ 
    *Modify a given time array so that:
    1) Initial transient regime is removed.
    2) Truncate signal end so as to have a whole number of periods.
    3) Use a constant time step (average time step).*

    :param time_array
    :type time_array: array
    :param frequency
    :type frequency: float
    :param new_initial_time
    :type new_initial_time: float
    :param trim_end: By default, the beginning of the time array is trimmed to have a 
    whole number of periods. You can choose to trim the end setting this parameter to *True*
    :type trim_end: bool

    :returns:
        - **clean_time_array** - rearranged array
    """

    # Remove a portion of initial time specified by user
    new_time_array = remove_initial_transient(time_array, new_initial_time)

    # Compute total time and average time step
    time_step = np.mean(new_time_array[1:] - new_time_array[:-1])
    total_time = new_time_array[-1] - new_time_array[0] + time_step

    # Determine new initial time so as to obtain a whole number of periods of
    # the target frequency
    nb_periods = int(round(frequency * total_time, ROUND_PRECISION))
    end_time = new_time_array[-1]
    initial_time = end_time - nb_periods / frequency + time_step

    # Do the contrary if trim_end is activated
    if trim_end:
        initial_time = time_array[0]
        end_time = initial_time + nb_periods / frequency - time_step

    # Reconstruct clean time array
    nb_samples = int(round((end_time - initial_time + time_step) / time_step, ROUND_PRECISION))
    clean_time_array = np.linspace(initial_time, end_time, nb_samples)

    return clean_time_array


def interpolate_signals(time_array, signal_array, ref_time_array):
    """ 
    *Interpolate a single signal on a reference time array.
    Interpolation is performed with scipy.interpolate.*

    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param ref_time_array
    :type ref_time_array: array
    
    :returns:
        - **new_signal** - Signal interpolated on ref_time_array
    """

    # Check input dimensions
    _check_input_dimensions(time_array, signal_array)

    # Deal with precision problems at bounds
    # (which can cause interpolate_function to mistakenly crash)
    if np.allclose(time_array[0], ref_time_array[0]):
        ref_time_array[0] = time_array[0]
    if np.allclose(time_array[-1], ref_time_array[-1]):
        ref_time_array[-1] = time_array[-1]

    # Simple linear interpolation
    interpolate_function = scipy.interpolate.interp1d(
        time_array, signal_array, axis=-1)  # , bounds_error=False)
    new_signal = interpolate_function(ref_time_array)

    return new_signal


def _check_input_dimensions(time_array, signal_array):
    """
    *Check that time_array and signal_array have the correct dimensions.
     - time_array should be 1D.
     - signal_array can be 1D or 2D but the size of the last dimension
       should be equal to the size of time_array*

    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    """

    if time_array.ndim != 1:
        raise ValueError('Time array should be 1-D. '
                         'Instead, dimension is {0}'.format(time_array.ndim))

    if signal_array.ndim not in [1, 2]:
        raise ValueError('Signal array should be 1-D or 2-D. '
                         'Instead, dimension is {0}.'.format(signal_array.ndim))

    if time_array.size != signal_array.shape[-1]:
        time_size = time_array.size
        signal_size = signal_array.shape[-1]
        raise ValueError('Last dimension of signal array should be equal to '
                         'the size of time array: size of time array ={0} ;'
                         ' size of signal array={1}'.format(time_size,
                                                            signal_size))

    return


def get_clean_signals(time_array, signal_array, target_frequency, transient_time=0.0):
    """ 
    *Truncate signals to get a whole number of periods of the target
    frequency.*
    
    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float

    :returns:
        - **clean_time_array** - Time array capturing a full number of periods according to the target_frequency
        - **clean_signal_array** - Signal interpolated on clean_time_array
    """

    # Check the dimensions of the inputs
    _check_input_dimensions(time_array, signal_array)

    # Trim time array to have a clean fft
    clean_time_array = define_good_time_array(
        time_array, target_frequency, transient_time)
    # Interpolate the signal
    clean_signal_array = interpolate_signals(
        time_array, signal_array, clean_time_array)

    return clean_time_array, clean_signal_array


def get_clean_fft(time_array, signal_array,
                  target_frequency, transient_time=0.0):
    """ 
    *Perform a clean Fast Fourier Transformation.
    Special care is taken of the time array to ensure a good definition of
    the Fourier Coefficient at the target_frequency.
    All input signals must have the same length and be defined
    on time_array.*

    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    
    :returns:
        - **fft_frequencies** - Array of fequencies
        - **fft_spectrum** - Array of fourier amplitudes
        - **index_target** - Integer, position of the value of the spectrum amplitude at the target frequency
    """

    clean_time_array, clean_signal_array = get_clean_signals(time_array,
                                                             signal_array,
                                                             target_frequency,
                                                             transient_time)

    # Perform Fourier Transformation and normalize it correctly, in
    # particular for value at 0 Hz.
    nb_samples = clean_time_array.size
    time_step = clean_time_array[1] - clean_time_array[0]
    zero_padding = 1  # Integer >= 1, (no zero-padding if 1)
    if zero_padding > 1:
        print(" ")
        print("--------------------------------------------------")
        print(" Zero padding used to display Fourier spectrum !")
        print(((" Frequency resolution of the plot: "
               "{0:10.5f} Hz").format(1. / zero_padding / time_step)))
        print("--------------------------------------------------")
    fft_spectrum = 2. * np.fft.rfft(clean_signal_array,
                                    n=zero_padding * nb_samples,
                                    axis=-1) / nb_samples
    if fft_spectrum.ndim == 1:
        fft_spectrum[0] /= 2.0
    else:
        fft_spectrum[:, 0] /= 2.0

    # Get the frequency array
    fft_frequencies = np.fft.rfftfreq(zero_padding * nb_samples, time_step)

    # Get the index of the element corresponding to target_frequency
    index_target = int(round(target_frequency * nb_samples * time_step, ROUND_PRECISION))
    # Not very satisfied with this... Should probably move this elsewhere

    return fft_frequencies, fft_spectrum, index_target


def get_coeff_fourier(time_array, signal_array,
                      target_frequency, transient_time=0.0):
    """ 
    *Computes the Discrete Fourier Coefficient at a single target frequency.*

    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    
    :returns:
        - **fourier_coefficient** - Array of the spectrum amplitude(s) at the target frequency
    """

    # Compute complete FFT (might be uselessly long for large arrays...)
    _, fft_spectrum, index_target = get_clean_fft(
        time_array, signal_array, target_frequency, transient_time)

    # Isolate the Fourier coefficient at target_frequency
    if fft_spectrum.ndim == 1:
        fourier_coefficient = fft_spectrum[index_target]
    else:
        fourier_coefficient = fft_spectrum[:, index_target]

    return fourier_coefficient


def get_clean_average(time_array, signal_array,
                      target_frequency, transient_time=0.0):
    """ 
    *Computes the average over a clean time array*
    
    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    
    :returns:
        - **average** - Float (or array), average(s) of the initial signal(s)
    """
    _, fft_spectrum, _ = get_clean_fft(
        time_array, signal_array, target_frequency, transient_time)
    if fft_spectrum.ndim == 1:
        average = np.abs(fft_spectrum[0])
    else:
        average = np.abs(fft_spectrum[:, 0])

    return average


def remove_average(signal):
    """ 
    *Remove temporal average from numpy array signal.
    This array can be multi-dimensional and averaging is
    performed over the last dimension. Make sure the signal is clean (dt is constant)*
    
    :param signal
    :type signal: array

    :returns:
        - **new_signal** - Array, signal with the average taken out
    """

    new_signal = signal - signal.mean(axis=-1, keepdims=True)

    return new_signal


def get_clean_psd(time_array, signal_array,
                  target_frequency, transient_time=0.0,
                  nb_blocks=1, window_func="flat", overlap_pct=0.0):
    """ 
    *Compute a clean Power Spectral Density with a time_array trimmed to
    suit target_frequency.
    Equivalent to matplotlib.mlab.psd with NFFT = number of samples and no
    windowing.*
    
    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param nb_blocks
    :type nb_blocks: int
    :param window_func
    :type window_func: string
    :param overlap_pct
    :type overlap_pct: float

    :returns: 
        - **fft_frequencies** - Array of frequencies of the PSD
        - **psd_spectrum** - Array of values of the PSD spectrum
        - **index_target** - Integer, position of the target frequency in fft_frequencies and psd_spectrum
    """

    try:
        import matplotlib.mlab as mlab

        # If more than one signal is used, matplotlib.mlab.psd does not work
        if signal_array.ndim > 1:
            raise ValueError

        time_array, signal_array = get_clean_signals(time_array,
                                                     signal_array,
                                                     target_frequency,
                                                     transient_time)
        sampling_frequency = 1. / (time_array[1] - time_array[0])

        block_size = int(len(signal_array) / nb_blocks)
        if len(signal_array) % nb_blocks != 0:
            print("WARNING : The number of windows requested for the PSD does " \
                  "not divide the total number of samples !!!")
            print("The frequency array might not contain exactly the target " \
                  "frequency anymore !")

        window_dict = {"flat": np.ones,
                       "hanning": np.hanning,
                       "hamming": np.hamming,
                       "blackman": np.blackman,
                       "bartlett": np.bartlett}

        window_mask = window_dict[window_func](block_size)

        psd_spectrum, fft_frequencies = mlab.psd(signal_array,
                                                 Fs=sampling_frequency,
                                                 NFFT=block_size,
                                                 window=window_mask,
                                                 detrend="none",  # Do not remove average trend
                                                 scale_by_freq="True",  # Divide by freq. resolution
                                                 noverlap=int(overlap_pct * block_size)
                                                 )

        index_target = int(round(target_frequency * block_size / sampling_frequency))

    except ImportError:
        print("matplotlib.mlab package is not available.")
        print("PSD is computed with no windowing.")
        fft_frequencies, fft_spectrum, index_target = get_clean_fft(
            time_array, signal_array, target_frequency, transient_time)

        frequency_resolution = fft_frequencies[1] - fft_frequencies[0]

        # Compute PSD with the correct normalization
        psd_spectrum = np.abs(fft_spectrum) ** 2 / (2. * frequency_resolution)
        if psd_spectrum.ndim == 1:
            psd_spectrum[0] *= 2
        else:
            psd_spectrum[:, 0] *= 2.

    except ValueError:
        fft_frequencies, fft_spectrum, index_target = get_clean_fft(
            time_array, signal_array, target_frequency, transient_time)

        frequency_resolution = fft_frequencies[1] - fft_frequencies[0]

        # Compute PSD with the correct normalization
        psd_spectrum = np.abs(fft_spectrum) ** 2 / (2. * frequency_resolution)
        if psd_spectrum.ndim == 1:
            psd_spectrum[0] *= 2
        else:
            psd_spectrum[:, 0] *= 2.

    return fft_frequencies, psd_spectrum, index_target


def get_average_psd(time, signals, target_frequency, transient_time, discardphase=False, nb_blocks=1,
                    window_func="flat", overlap=0.0):
    """ 
    *Compute the average PSD of several signals and check the Parseval equality.
    Two options are possible: with/ without Fourier phases.*
    
    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param discradphase
    :type discardphase: bool
    :param nb_blocks
    :type nb_blocks: int
    :param window_func
    :type window_func: string
    :param overlap
    :type overlap: float

    :returns:
        - **psd_average** -
        - **psd_freqs** -
        - **index_target** -
        - **energy_time** -
        - **energy_freq** -
    """
    timestep = time[1] - time[0]
    nb_samples = len(time)

    if discardphase:
        psd_freqs, psd_all, index_target = get_clean_psd(time, signals, target_frequency, transient_time,
                                                         nb_blocks, window_func, overlap)
        psd_average = np.average(psd_all, axis=0)  # Average over all probes
        energy_time = np.sum(signals ** 2, axis=-1) / nb_samples
        energy_time = np.average(energy_time)
    else:
        average = np.average(signals, axis=0)  # Average over all probes
        psd_average, psd_freqs, index_target = get_clean_psd(time, average, target_frequency, transient_time,
                                                             nb_blocks, window_func, overlap)
        energy_time = np.sum(average ** 2, axis=-1) / nb_samples

    freqstep = psd_freqs[1] - psd_freqs[0]

    energy_freq = np.sum(psd_average * freqstep, axis=-1)

    return psd_average, psd_freqs, index_target, energy_time, energy_freq


def apply_butter_bandpass_filter(time_array, signal_array, low_frequency,
                                 high_frequency, order=3):
    """ 
    *Apply bandpass filtering with a Butterworth filter.*

    """

    def _get_butter_bandpass_parameters(
            low_frequency, high_frequency, sampling_frequency, order=5):
        """ 
        *Define the parameters of the Butterworth bandpass filter*
        """
        nyquist_frequency = 0.5 * sampling_frequency
        frequency_limits = [low_frequency / nyquist_frequency,
                            high_frequency / nyquist_frequency]
        butter_b, butter_a = scipy.signal.butter(
            order, frequency_limits, btype='band')
        return butter_b, butter_a

    time_step = np.mean(time_array[1:] - time_array[:-1])
    sampling_frequency = 1. / time_step
    butter_b, butter_a = _get_butter_bandpass_parameters(low_frequency,
                                                         high_frequency,
                                                         sampling_frequency,
                                                         order=order)
    # Use forward-backward filter improves filtered reconstruction at signal
    # ends.
    try: 
        filtered_signal = scipy.signal.filtfilt(
            butter_b, butter_a, signal_array, axis=-1, method='gust')
    except TypeError: # If "method" is not implemented in scipy version
        filtered_signal = scipy.signal.filtfilt(
            butter_b, butter_a, signal_array, axis=-1)

    return filtered_signal


def compute_cumulative_fourier(time_array, signal_array, target_frequency, 
                               transient_time=0.0):
    """
    *Compute moving Fourier Coefficient at the target frequency,
    for a set of signals*
    
    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float

    :returns:
        - **cumul_fourier** - Cumulative fourier coefficient over increasing number of periods
        - **nb_of_periods** - Cumulative number of periods
    """

    time, signals = get_clean_signals(time, signals, target_frequency, transient_time)
    total_time = time[-1] + time[1] - 2 * time[0]
    nb_periods = int(round(target_frequency * total_time, ROUND_PRECISION))

    if signals.ndim == 1:
        cumul_fourier_coeffs = 1.j * np.zeros(nb_periods)
        for id_period in range(nb_periods):
            max_time = time[0] + float(id_period + 1) / target_frequency
            cumul_time = time[time < max_time]
            cumul_signals = signals[time < max_time]
            cumul_fourier_coeffs[id_period] = get_coeff_fourier(
                cumul_time, cumul_signals, target_frequency, transient_time=0.0
            )
    elif signals.ndim > 1:
        cumul_fourier_coeffs = 1.j * np.zeros((signals.shape[0], nb_periods))
        for id_period in range(nb_periods):
            max_time = time[0] + float(id_period + 1) / target_frequency
            cumul_time = time[time < max_time]
            cumul_signals = signals[:, time < max_time]
            cumul_fourier_coeffs[:, id_period] = get_coeff_fourier(
                cumul_time, cumul_signals, target_frequency, transient_time=0.0
            )

    return cumul_fourier_coeffs, np.arange(1, nb_periods + 1, 1)


def get_psd_harmonics(time_array, signal_array,
                      target_frequency, transient_time=0.0,
                      nb_blocks=1, window_shape="flat", overlap=0.0):
    """ 
    *Compute the PSD of signal fluctuations and sort out the
    contributions of target_frequency, first and second harmonic
    from noise.
    TODO : Make sure this can read multiple signals.*

    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float
    :param window_func
    :type window_func: string
    :param overlap
    :type overlap: float

    :returns:
        - **psd_contributions** - Array containing the fraction of the PSD contained in the target frequency, first, second harmony and in the rest of the spectrum respectively
    """

    # Extract fluctuations
    signal_array = remove_average(signal_array)

    # Obtain the PSD:
    psd_spectrum, psd_frequencies, index_target = get_clean_psd(
        time_array, signal_array, target_frequency, transient_time,
        nb_blocks, window_shape, overlap
    )

    # Separate the contributions from the target frequency, first, second
    # harmonic and rest (noise)
    frequency_step = psd_frequencies[1] - psd_frequencies[0]
    if psd_spectrum.ndim == 1:
        total_power = np.sum(psd_spectrum * frequency_step)
        psd_contributions = np.zeros(4)
        # Zeroth harmonic (target_frequency)
        psd_contributions[1] = psd_spectrum[
                                   index_target] * frequency_step / total_power
        # First harmonic
        psd_contributions[2] = psd_spectrum[
                                   2 * index_target] * frequency_step / total_power
        # Second harmonic
        psd_contributions[3] = psd_spectrum[
                                   3 * index_target] * frequency_step / total_power
        # Rest
        psd_contributions[0] = 1 - np.sum(psd_contributions)
    else:
        nb_signals = signal.shape[0]
        total_power = np.sum(psd_spectrum * frequency_step, axis=-1)
        psd_contributions = np.zeros((nb_signals, 4))
        # Zeroth harmonic (target_frequency)
        psd_contributions[:, 1] = psd_spectrum[:,
                                  index_target] * frequency_step / total_power
        # First harmonic
        psd_contributions[:, 2] = psd_spectrum[:,
                                  2 * index_target] * frequency_step / total_power
        # Second harmonic
        psd_contributions[:, 3] = psd_spectrum[:,
                                  3 * index_target] * frequency_step / total_power
        # Rest
        psd_contributions[:, 0] = 1 - np.sum(psd_contributions, axis=-1)

    return psd_contributions


def get_cumulative_average(time_array, signal_array, target_frequency, transient_time=0.0):
    """
    *Compute the cumulative moving average over a series of signals
    stored as rows of the numpy array "signals".
    For now, samples set for averaging is increased one sample at
    a time.
    TO DO : allow for a different "speed" of moving average.*

    :param time_array
    :type time_array: array
    :param signal_array
    :type signal_array: array
    :param target_frequency
    :type target_frequency: float
    :param transient_time
    :type transient_time: float

    :returns:
        - **cumul_average** - Array containing the cumulative average of the signal over an increasing number of periods
    """

    # number of periods
    total_time = time_array[-1] + time_array[1] - 2 * time_array[0]
    nb_periods = int(round(target_frequency * total_time, ROUND_PRECISION))
    sampling_frequency = 1. / (time_array[1] - time_array[0])
    samples_per_period = int(round(sampling_frequency / target_frequency, ROUND_PRECISION))
    start = int(round(max(0.0, transient_time - time_array[0]) * sampling_frequency, ROUND_PRECISION))

    timestep_tol = 3
    # Array is going to be truncated to the right size anyway.
    if signal_array.ndim == 1:
        cumul_average = np.zeros(nb_periods)
        for id_period in range(nb_periods):
            end = start + min((id_period + 1) *
                              samples_per_period + timestep_tol, time_array.size)
            cumul_signals = signal_array[start:end]
            cumul_average[id_period] = np.mean(cumul_signals)
    elif signal_array.ndim > 1:
        cumul_average = (
            np.zeros((signal_array.shape[0], nb_periods)))
        for id_period in range(nb_periods):
            end = min(
                (id_period + 1) * samples_per_period + timestep_tol,
                time_array.size)
            cumul_signals = signal_array[
                            :, start:end]
            cumul_average[:, id_period] = np.average(cumul_signals, axis=-1)

    return np.squeeze(cumul_average)
