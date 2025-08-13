import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import scipy.stats as stats


def resample_signal(time, signal, dtime=None):
    """
	*Resample the initial signal at a constant time interval.*
  
	:param time: Time vector of your signal
	:param signal: Signal vector
	:dtime: New time step
	:returns:

	- **rescaled_time** - Uniformally rescaled time vector
	- **rescaled_signal** - Rescaled signal

	.. note:: 	If a dtime is given, the interpolation is made to 
				have a signal with a time interval of dtime.
				Else, the dt is the smallest time interval between
				two values of the signal.
	"""

    if dtime is None:
        time_steps = np.diff(time)
        dtime = time_steps.min()

    rescaled_time = time - time[0]
    rescaled_time = np.arange(rescaled_time[0], rescaled_time[-1], dtime)

    rescaled_signal = np.interp(rescaled_time, time - time[0], signal)

    return rescaled_time, rescaled_signal


def calc_autocorrelation_time(time, signal, threshold=0.2):
    """
	*Estimate the autocorrelation time at a given threshold.*
	
	:param time: Time vector of your signal
	:param signal: Signal vector
	:param threshold: Threshold under which the signal is correlated
	:type threshold: float

	:returns:

	- **autocorrelation_time** - Minimum time step to capture the signal at a
						correlation under the threshold
	"""

    fluctu = signal - np.mean(signal)
    time -= time[0]
    correlation = np.correlate(fluctu, fluctu, "full")
    correlation = abs(correlation[int((correlation.size - 1) / 2) :])
    correlation /= correlation[0]
    # 1-D interpol on correlation
    # scipy interp1d is probably overkill 
    idx = max(np.argmax(correlation < threshold), 1)
    low_cor = correlation[idx]
    up_cor = correlation[idx-1]
    alpha = (up_cor-threshold)/(up_cor-low_cor)
    autocorrelation_time = (
        alpha*time[idx]
        + (1.-alpha)*time[idx-1]
    )
    # #autocorrelation_time = time[np.argmax(correlation < threshold)] 
    # plt.plot(time, signal ,"+")
    # plt.plot(time, correlation, "o")
    # plt.plot([autocorrelation_time,autocorrelation_time], [0,1.])
    # plt.plot([0,1], [threshold,threshold])
    # plt.show()


    return autocorrelation_time


def show_autocorrelation_time(time, signal, threshold=0.2):
    """
	*Plot the autocorrelation function of the signal.*
	
	:param time: Time vector of your signal
	:param signal: Signal vector
	:param threshold: Autocorrelation threshold

	:returns:

	- **fig** - Figure of the result

	"""

    signal -= np.mean(signal)
    correlation = np.correlate(signal, signal, "full")
    correlation = abs(correlation[int((correlation.size - 1) / 2) :])
    correlation /= correlation[0]

    autocorrelation_time = time[np.argmax(correlation < threshold)]

    figout = plt.figure()
    plt.plot(time, correlation)
    plt.axvline(autocorrelation_time, c="r")
    plt.xlabel("Time (s)")
    plt.ylabel("Correlation")
    plt.semilogx()
    plt.grid()

    return figout


def sort_spectral_power(time, signal):
    """
	*Determine the harmonic power contribution of the signal.*
	
	:param time: Time vector of your signal
	:param signal: Signal vector

	returns:

	- **harmonic_power** - Harmonic power of the signal
	- **total_power** - Total spectral power of the signal


	.. note:: 	It calculates the Power Spectral Density (PSD) of the complete
				signal and of a downsampled version of the signal. The 
				difference of the two PSD contains only harmonic components.
	"""

    frequency1, psd1 = power_spectral_density(time, signal)
    deltaf1 = 1.0 / ((time[1] - time[0]) * 2)

    downsampled_time = time[0:-1:2]
    downsampled_signal = signal[0:-1:2]
    frequency2, psd2 = power_spectral_density(downsampled_time, downsampled_signal)
    deltaf2 = 1.0 / ((downsampled_time[1] - downsampled_time[0]) * 2)

    psd2 = np.interp(frequency1, frequency2, psd2) * 2

    harmonic_power = np.cumsum(np.abs(psd1 - psd2) * 2)[-1] * deltaf1
    total_power = np.cumsum(psd1)[-1] * deltaf1

    return harmonic_power, total_power


def power_representative_frequency(time, signal, threshold=0.8):
    """
	*Calculate the frequency that captures a level of spectral power.*

	:param time: Time vector of your signal
	:param signal: Signal vector
	:param threshold: Level of representativity of the spectral power

	:returns:

	- **representative_frequency** - Frequency above which the power threshold is reached

	.. note:: 	It calculates the cumulative power spectral density and returns
				the frequency that reaches the threshold of spectral power.
	"""

    frequency, power_density = power_spectral_density(time, signal)

    cumulative_psd = np.cumsum(power_density)
    cumulative_psd /= cumulative_psd[-1]
    representative_frequency = frequency[np.argmax(cumulative_psd > threshold)]

    return representative_frequency


def show_power_representative_frequency(time, signal, threshold=0.80):
    """
	*Plot the power spectral density of the signal.*
	
	:param time: Time vector of your signal
	:param signal: Signal vector
	:param threshold: Power representative frequency threshold

	:returns:

	- **fig** - Figure of the result

	"""

    frequency, power_density = power_spectral_density(time, signal)
    cumulative_psd = np.cumsum(power_density)
    cumulative_psd /= cumulative_psd[-1]
    representative_frequency = frequency[np.argmax(cumulative_psd > threshold)]

    figout = plt.figure()
    plt.plot(frequency, cumulative_psd)
    plt.axvline(representative_frequency, c="r")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("PSD (%)")
    plt.grid()

    return figout


def ks_test_distrib(data, distribution="normal"):
    """
	*Calculate the correlation score of the signal with the distribution*

	:param data: array of values
	:param distribution: kind of distribution the values follow to test

	:returns:

	- **score** -Minimum score over the height of the ks test
	- **position** -Index of the height at which the min. of the test is found
	- **height** -Corresponding heiht where the min. is found
	- **scale** -Scale parameter of the lognormal fitting

	"""
    data_shape = np.shape(data)
    if len(data_shape) > 1:
        p_value = np.zeros(data_shape[1])
        for j in range(data_shape[1]):
            data_1d = data[:, j]

            mu = np.average(data_1d)
            sigma = np.std(data_1d)

            # if data_1d.min() <= 0:
            # 	p_value[j]=1
            # else:
            if distribution == "lognormal":
                sigma, loc, scale = stats.lognorm.fit(data_1d, floc=0)
                stat, p_value[j] = stats.kstest(
                    data_1d, "lognorm", args=(sigma, loc, scale)
                )
                # if p_value[j]<5.0e-2:
                # 	print(p_value[j])
                # 	print(temperature.min())
                # 	print(temperature.max())

            elif distribution == "normal":
                loc, sigma = stats.norm.fit(data_1d, loc=300)
                stat, p_value[j] = stats.kstest(data_1d, "norm", args=(loc, sigma))
                scale = None
        position = np.argmin(p_value)
        height = position / data_shape[1]
    else:
        p_value = 0

        data_1d = data
        mu = np.average(data_1d)
        sigma = np.std(data_1d)
        if distribution == "lognormal":
            sigma, loc, scale = stats.lognorm.fit(data_1d, floc=0)
            stat, p_value = stats.kstest(data_1d, "lognorm", args=(sigma, loc, scale))
            # if p_value[j]<5.0e-2:
            # 	print(p_value[j])
            # 	print(temperature.min())
            # 	print(temperature.max())

        elif distribution == "normal":
            loc, sigma = stats.norm.fit(data_1d, loc=300)
            stat, p_value = stats.kstest(data_1d, "norm", args=(loc, sigma))
            scale = None
        position = np.argmin(p_value)
        height = None

    # print(p_value)
    score = np.min(p_value)

    return score, position, height, scale


def show_temperature_distribution(temperature_recording, height, distribution="normal"):
    """
	*Plot the temperature distribution and the fitting curve*

	:param temperature_recording: Temperature as a function of height and time
	:param height: Height in the plan40
	:param distribution: Type of distribution for the fitting method

	:returns:

	- **fig** - Figure of the result
	"""
    hH = np.linspace(0, 1, len(temperature_recording[0, :]))
    index = np.abs(hH - height).argmin()
    data = temperature_recording[:, index]

    abscissa = np.linspace(0, np.amax(data), num=50)

    if distribution == "normal":
        loc, sigma = stats.norm.fit(data, loc=300)
        pdf_fitted = stats.norm.pdf(abscissa, loc, sigma)

    if distribution == "lognormal":
        sigma, loc, scale = stats.lognorm.fit(data, floc=300)
        pdf_fitted = stats.lognorm.pdf(abscissa, sigma, loc, scale)

    figout = plt.figure()
    plt.hist(data, density=True, bins=30, color="lightgray")
    plt.plot(abscissa, pdf_fitted)
    plt.xlabel("T (K)")

    return figout


def duration_for_uncertainty(
    time, signal, target=10, confidence=0.95, distribution="normal"
):
    """
	*Give suggestion of simulation duration of a plan40 calculation.*
	
	:param time: Time vector of your signal
	:param signal: Signal vector
	:param target: Desired amplitude of the confidence interval
	:param confidence: Level of confidence of the interval
	:param distribution: Type of distribution of the signal to make the interval

	:returns:

	- **duration** -Duration of the signal
	"""

    dtime = calc_autocorrelation_time(time, signal)
    sigma = np.std(signal)

    if distribution == "normal":
        z_coef = stats.norm.ppf(1 - (1 - confidence) / 2)
        n_timesteps = int((2 * z_coef * sigma / target) ** 2)

    duration = dtime * n_timesteps

    return duration


def uncertainty_from_duration(
    dtime, sigma, duration, confidence=0.95, distribution="normal"
):
    """
	*Give confidence interval length of a plan40 calculation.*

	:param dtime: Time step of your solutions
	:param sigma: Standard deviation of your signal
	:param duration: Desired duration of the signal
	:param confidence: Level of confidence of the interval
	:param distribution: Type of distribution of the signal to make the interval

	:returns:

	- **length** - Length of the confidence interval in K
	"""

    if distribution == "normal":
        z_coef = stats.norm.ppf(1 - (1 - confidence) / 2)
        # print(z_coef)
        n_timesteps = int(duration / dtime)
        length = 2 * z_coef * sigma / np.sqrt(n_timesteps)

    return length


def convergence_cartography(time, signal, **kwargs):
    """
	*Create a cartography of the convergence of the confidence interval in a simulation.*

	:param time: Time vector of your signal
	:param signal: Signal vector


	==**kwargs==
	
		:param max_time: Maximal simulation duration
		:param interlen: Maximal interval length

	:returns:
		- **fig** - Figure of the cartography

	"""
    max_time = kwargs.get("max_time", None)
    interlen = kwargs.get("interlen", None)

    if max_time is None:
        max_time = 0.05
    if interlen is None:
        interlen = 50

    dtime = calc_autocorrelation_time(time, signal)
    sigma = np.std(signal)
    durations = np.arange(0.001, max_time, 0.001)
    interval_lengths = np.arange(0, interlen)
    level = np.zeros((len(durations), len(interval_lengths)))
    for idx, dur in enumerate(durations):
        for idy, length in enumerate(interval_lengths):
            sqrtn = int(np.sqrt(dur / dtime))
            z_star = length * sqrtn / (2 * sigma)
            dec_per = stats.norm.cdf(z_star)
            level[idx, idy] = 1 - (1 - dec_per) * 2

    fig = plt.figure()
    fig = plt.contour(
        durations, interval_lengths, np.transpose(level), [0.6, 0.8, 0.9, 0.95, 0.975]
    )
    # plt.legend()
    plt.xlabel("Simulation duration (s)")
    plt.ylabel("Interval amplitude (K)")
    plt.clabel(
        fig,
        inline=1,
        fontsize=10,
        colors="k",
        fmt={0.6: "60%", 0.8: "80%", 0.9: "90%", 0.95: "95%", 0.975: "97.5%"},
    )
    plt.title("Contour of confidence interval(simulation duration, interval amplitude)")
    plt.grid(True)
    return fig


def calculate_std(time, signal, frequency):
    """
	*Give the standard deviation of a signal at a given frequency.*
	
	:param time: Time vector of your signal
	:param signal: Signal vector
	:param frequency: Frequency at which values of the recording are taken

	:returns:

		- **std** - Standard deviation of the values taken from the recording
	"""

    dtime = 1 / frequency
    time -= time[0]
    rescaled_time = np.arange(0, time[-1], dtime)
    rescaled_signal = np.interp(rescaled_time, time, signal)

    std = np.std(rescaled_signal)

    return std


def power_spectral_density(time, signal):
    """
	*Automate the computation of the Power Spectral Density of a signal.*

	:param time: Time vector of your signal
	:param signal: Signal vector

	:returns:

	- **frequency** -Frequency vector of the signal's power spectral density
	- **power_spectral_density** -Power spectral density of the signal
	"""

    total_time = time[-1] - time[0]
    dtime = time[1] - time[0]

    freq_resolution = 1.0 / total_time
    freq_nyquist = 1.0 / dtime

    N = len(time)

    frequency = np.arange(0, freq_nyquist, freq_resolution, dtype=np.float)
    frequency = frequency[: int(N / 2) - 1]

    mean_amplitude = np.mean(signal)

    raw_fft = np.fft.fft(signal - mean_amplitude)
    power_spectral_density = np.square(
        np.absolute(raw_fft)[: int(N / 2) - 1] / np.sqrt(N)
    )
    power_spectral_density[1:] *= 2

    return frequency, power_spectral_density


def to_percent(y, position):
    """
	*Rescale the y-axis to per*
    
    """
    s = str(round(100.0 * y))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams["text.usetex"] is True:
        return s + r"$\%$"
    else:
        return s + "%"


def plot_distributions(path="./data.txt"):
    formatter = FuncFormatter(to_percent)
    data = np.genfromtxt(path)
    S = np.shape(data)

    t_fusion = 1930.0
    bin_edges = list(np.linspace(t_fusion - 200.0, t_fusion + 200.0, 21))

    for j in range(S[1]):

        temperatures = data[:, j]

        clipped_temperatures = np.clip(temperatures, bin_edges[0], bin_edges[-1])
        clipped_temperatures.sort()
        weights = np.ones_like(clipped_temperatures) / float(len(clipped_temperatures))
        lower = 0
        upper = -1
        lower = np.argmax(clipped_temperatures > bin_edges[0]) - 1
        upper = np.argmax(clipped_temperatures == bin_edges[-1])

        if upper == 0:
            upper = -1

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, sharey=True, gridspec_kw={"width_ratios": [1, 5, 1]}
        )

        rect = plt.Rectangle((0.4, 0), 0.2, np.sum(weights[: lower + 1]))
        ax1.add_patch(rect)
        plt.sca(ax1)
        plt.xticks([0.5], ["<" + str(t_fusion - 200)])
        plt.gca().yaxis.set_major_formatter(formatter)
        ax1.set_ylabel("Probability")

        ax2.hist(
            clipped_temperatures[lower:upper],
            bins=bin_edges,
            weights=weights[lower:upper],
        )
        xticks = list(np.linspace(t_fusion - 200.0, t_fusion + 200.0, 11))
        xlabels = list(map(str, xticks))
        plt.sca(ax2)
        plt.xticks(xticks, xlabels, rotation=45, size=8)
        plt.gca().yaxis.set_major_formatter(formatter)

        rect = matplotlib.patches.Rectangle((0.4, 0), 0.2, np.sum(weights[upper:]))
        ax3.add_patch(rect)
        plt.sca(ax3)
        plt.xticks([0.5], [str(t_fusion + 200) + "<"])
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.savefig("pdf_h_%s.png" % j)
        plt.close()
