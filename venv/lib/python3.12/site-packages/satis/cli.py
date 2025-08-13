import click
from satis.io import (read_signals, extract_subset, write_output, plot_time_signals,
					   plot_fourier_convergence, plot_psd_convergence, plot_fourier_variability, plot_psd_variability)
from satis.avbp2global import display_header
import os
import pkg_resources
import matplotlib.pyplot as plt


@click.group()
def main_cli():
    """---------------   SATIS  --------------------

You are now using the Command line interface of Satis,
a simple tool for spectral analysis optimized to signals from CFD,
created at CERFACS (https://cerfacs.fr).

This is a python package currently installed in your python environment.
"""
    pass

## Time
@click.command()
@click.argument("filename",
				default='signals.dat',
				type=str)
@click.option("--freq", "-f", "target_frequency", type=float, default=0.0, required=False,
			  help="Set the target frequency in Hz.")
@click.option("--discardphase", is_flag=True, help="Set all Fourier phases to zero when computing the PSD.")
@click.option("--dbspl", is_flag=True,
			  help="Display psd_convergence plot in dB SPL (valid only for pressure fluctuations).")
@click.option("--unit", type=str, default="Unit", help="Unit of the input signals (for plots).")
@click.option("--tinit", "-t", "transient_time", type=float, default=0.0, required=False,
			  help='Remove an initial portion of '
				   'the signal in s (optional). '
				   'Using 0.0 deactivates the filter.')
@click.option("--window_blocks", "nb_blocks", type=int, default=1, required=False,
			  help="For PSD convergence only: number of windows. Default is 1.")
@click.option("--window_shape",
			  type=click.Choice(
			  ["flat",
				  "hanning",
				  "hamming",
				  "blackman",
				  "bartlett"
			  ],
			  case_sensitive=False),
			  default="flat",
			  help="For PSD convergence only: shape of the windowing function. Default is flat.")
@click.option("--window_overlap", "overlap", type=float, default=0.0, required=False,
			  help='For PSD convergence only: overlapping between consecutive windows. '
				   'Should be between 0 and 1. '
				   'Default is 0.0. ')
@click.option("--remove_average", is_flag=True,
			  help="Remove temporal average of all signals.")
@click.option("--subset", type=int, multiple=True, help="Use only a subset of signals.")
def time(filename, target_frequency, discardphase, dbspl, unit, transient_time, nb_blocks, window_shape,
			 overlap, remove_average, subset):
	"""Plot the temporal signal and its time-average.

	Plot the temporal signal, the ensemble average, and the cumulative
	 time-average.
	"""

	display_header()

	# TODO: virer
	# sig_data = open(filename, 'r')
	# sig_dict = yaml.safe_load(sig_data)
	# TIME, SIGNALS = read_signal_yaml(sig_dict)

	TIME, SIGNALS = read_signals(filename, remove_average)

	if subset:
	    SIGNALS = extract_subset(subset, SIGNALS)

	write_output(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		discardphase,
		unit,
		dbspl)

	fig = plot_time_signals(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		unit)

	plt.show()

main_cli.add_command(time)


## fourier_convergence
@click.command()
@click.argument("filename",
				default='signals.dat',
				type=str)
@click.option("--freq", "-f", "target_frequency", type=float, default=0.0, required=False,
			  help="Set the target frequency in Hz.")
@click.option("--discardphase", is_flag=True, help="Set all Fourier phases to zero when computing the PSD.")
@click.option("--dbspl", is_flag=True,
			  help="Display psd_convergence plot in dB SPL (valid only for pressure fluctuations).")
@click.option("--unit", type=str, default="Unit", help="Unit of the input signals (for plots).")
@click.option("--tinit", "-t", "transient_time", type=float, default=0.0, required=False,
			  help='Remove an initial portion of '
				   'the signal in s (optional). '
				   'Using 0.0 deactivates the filter.')
@click.option("--window_blocks", "nb_blocks", type=int, default=1, required=False,
			  help="For PSD convergence only: number of windows. Default is 1.")
@click.option("--window_shape",
			  type=click.Choice(
			  ["flat",
				  "hanning",
				  "hamming",
				  "blackman",
				  "bartlett"
			  ],
			  case_sensitive=False),
			  default="flat",
			  help="For PSD convergence only: shape of the windowing function. Default is flat.")
@click.option("--window_overlap", "overlap", type=float, default=0.0, required=False,
			  help='For PSD convergence only: overlapping between consecutive windows. '
				   'Should be between 0 and 1. '
				   'Default is 0.0. ')
@click.option("--remove_average", is_flag=True,
			  help="Remove temporal average of all signals.")
@click.option("--subset", type=int, multiple=True, help="Use only a subset of signals.")
def fourierconvergence(filename, target_frequency, discardphase, dbspl, unit, transient_time, nb_blocks, window_shape,
			 overlap, remove_average, subset):
	"""
	Plot discrete Fourier transform of the complete average signal

	The plot displays the DFT based on the whole signal, on its last half and its last quarter.
	Note that the "half" and "quarter" signals are made of a whole number of periods.
	WARNING!
	This diagnostic works on the ensemble average (average signal in "time" diagnostic).
	"""

	display_header()

	# sig_data = open(filename, 'r')
	# sig_dict = yaml.safe_load(sig_data)
	# TIME, SIGNALS = read_signal_yaml(sig_dict)

	TIME, SIGNALS = read_signals(filename, remove_average)

	if subset:
	    SIGNALS = extract_subset(subset, SIGNALS)

	write_output(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		discardphase,
		unit,
		dbspl)

	fig = plot_fourier_convergence(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		unit)

	plt.show()

main_cli.add_command(fourierconvergence)


## psd_convergence
@click.command()
@click.argument("filename",
				default='signals.dat',
				type=str)
@click.option("--freq", "-f", "target_frequency", type=float, default=0.0, required=False,
			  help="Set the target frequency in Hz.")
@click.option("--discardphase", is_flag=True, help="Set all Fourier phases to zero when computing the PSD.")
@click.option("--dbspl", is_flag=True,
			  help="Display psd_convergence plot in dB SPL (valid only for pressure fluctuations).")
@click.option("--unit", type=str, default="Unit", help="Unit of the input signals (for plots).")
@click.option("--tinit", "-t", "transient_time", type=float, default=0.0, required=False,
			  help='Remove an initial portion of '
				   'the signal in s (optional). '
				   'Using 0.0 deactivates the filter.')
@click.option("--window_blocks", "nb_blocks", type=int, default=1, required=False,
			  help="For PSD convergence only: number of windows. Default is 1.")
@click.option("--window_shape",
			  type=click.Choice(
			  ["flat",
				  "hanning",
				  "hamming",
				  "blackman",
				  "bartlett"
			  ],
			  case_sensitive=False),
			  default="flat",
			  help="For PSD convergence only: shape of the windowing function. Default is flat.")
@click.option("--window_overlap", "overlap", type=float, default=0.0, required=False,
			  help='For PSD convergence only: overlapping between consecutive windows. '
				   'Should be between 0 and 1. '
				   'Default is 0.0. ')
@click.option("--remove_average", is_flag=True,
			  help="Remove temporal average of all signals.")
@click.option("--subset", type=int, multiple=True, help="Use only a subset of signals.")
def psdconvergence(filename, target_frequency, discardphase, dbspl, unit, transient_time, nb_blocks, window_shape,
			 overlap, remove_average, subset):
	"""Plot the PSD convergence diagnostic results.

	The plot displays the PSD based on the whole signal, on its last half and its last quarter.
	Note that the "half" and "quarter" signals are made of a whole number of periods.
	WARNING!
	This diagnostic works on the ensemble average (average signal in "time" diagnostic).
	
	"""

	display_header()

	# sig_data = open(filename, 'r')
	# sig_dict = yaml.safe_load(sig_data)
	# TIME, SIGNALS = read_signal_yaml(sig_dict)

	TIME, SIGNALS = read_signals(filename, remove_average)

	if subset:
	    SIGNALS = extract_subset(subset, SIGNALS)

	write_output(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		discardphase,
		unit,
		dbspl)

	fig = plot_psd_convergence(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		discardphase,
		dbspl,
		unit,
		nb_blocks,
		window_shape,
		overlap)
	
	plt.show()

main_cli.add_command(psdconvergence)


## fourier_variability
@click.command()
@click.argument("filename",
				default='signals.dat',
				type=str)
@click.option("--freq", "-f", "target_frequency", type=float, default=0.0, required=False,
			  help="Set the target frequency in Hz.")
@click.option("--discardphase", is_flag=True, help="Set all Fourier phases to zero when computing the PSD.")
@click.option("--dbspl", is_flag=True,
			  help="Display psd_convergence plot in dB SPL (valid only for pressure fluctuations).")
@click.option("--unit", type=str, default="Unit", help="Unit of the input signals (for plots).")
@click.option("--tinit", "-t", "transient_time", type=float, default=0.0, required=False,
			  help='Remove an initial portion of '
				   'the signal in s (optional). '
				   'Using 0.0 deactivates the filter.')
@click.option("--window_blocks", "nb_blocks", type=int, default=1, required=False,
			  help="For PSD convergence only: number of windows. Default is 1.")
@click.option("--window_shape",
			  type=click.Choice(
			  ["flat",
				  "hanning",
				  "hamming",
				  "blackman",
				  "bartlett"
			  ],
			  case_sensitive=False),
			  default="flat",
			  help="For PSD convergence only: shape of the windowing function. Default is flat.")
@click.option("--window_overlap", "overlap", type=float, default=0.0, required=False,
			  help='For PSD convergence only: overlapping between consecutive windows. '
				   'Should be between 0 and 1. '
				   'Default is 0.0. ')
@click.option("--remove_average", is_flag=True,
			  help="Remove temporal average of all signals.")
@click.option("--subset", type=int, multiple=True, help="Use only a subset of signals.")
def fouriervariability(filename, target_frequency, discardphase, dbspl, unit, transient_time, nb_blocks, window_shape,
			 overlap, remove_average, subset):
	"""Plot the Fourier variability diagnostic results.

	For each signal, plot the mean and fluctuations around the mean 
	and the Fourier coefficient at the target frequency.
	"""

	display_header()

	# sig_data = open(filename, 'r')
	# sig_dict = yaml.safe_load(sig_data)
	# TIME, SIGNALS = read_signal_yaml(sig_dict)

	TIME, SIGNALS = read_signals(filename, remove_average)

	if subset:
	    SIGNALS = extract_subset(subset, SIGNALS)

	write_output(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		discardphase,
		unit,
		dbspl)

	fig = plot_fourier_variability(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		unit)

	plt.show()

main_cli.add_command(fouriervariability)


## psd_variability
@click.command()
@click.argument("filename",
				default='signals.dat',
				type=str)
@click.option("--freq", "-f", "target_frequency", type=float, default=0.0, required=False,
			  help="Set the target frequency in Hz.")
@click.option("--discardphase", is_flag=True, help="Set all Fourier phases to zero when computing the PSD.")
@click.option("--dbspl", is_flag=True,
			  help="Display psd_convergence plot in dB SPL (valid only for pressure fluctuations).")
@click.option("--unit", type=str, default="Unit", help="Unit of the input signals (for plots).")
@click.option("--tinit", "-t", "transient_time", type=float, default=0.0, required=False,
			  help='Remove an initial portion of '
				   'the signal in s (optional). '
				   'Using 0.0 deactivates the filter.')
@click.option("--window_blocks", "nb_blocks", type=int, default=1, required=False,
			  help="For PSD convergence only: number of windows. Default is 1.")
@click.option("--window_shape",
			  type=click.Choice(
			  ["flat",
				  "hanning",
				  "hamming",
				  "blackman",
				  "bartlett"
			  ],
			  case_sensitive=False),
			  default="flat",
			  help="For PSD convergence only: shape of the windowing function. Default is flat.")
@click.option("--window_overlap", "overlap", type=float, default=0.0, required=False,
			  help='For PSD convergence only: overlapping between consecutive windows. '
				   'Should be between 0 and 1. '
				   'Default is 0.0. ')
@click.option("--remove_average", is_flag=True,
			  help="Remove temporal average of all signals.")
@click.option("--subset", type=int, multiple=True, help="Use only a subset of signals.")
def psdvariability(filename, target_frequency, discardphase, dbspl, unit, transient_time, nb_blocks, window_shape,
			 overlap, remove_average, subset):
	"""Plot the spectral energy at the target frequency.

	Plot the spectral energy of the first and second harmonic of the target frequency.
	Note that  the distribution is related to the fluctuation, thus the time average 
	has been removed from the signal.
	"""

	display_header()

	# sig_data = open(filename, 'r')
	# sig_dict = yaml.safe_load(sig_data)
	# TIME, SIGNALS = read_signal_yaml(sig_dict)

	TIME, SIGNALS = read_signals(filename, remove_average)

	if subset:
	    SIGNALS = extract_subset(subset, SIGNALS)

	write_output(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		discardphase,
		unit,
		dbspl)

	fig = plot_psd_variability(
		TIME,
		SIGNALS,
		target_frequency,
		transient_time,
		nb_blocks,
		window_shape,
		overlap)

	plt.show()

main_cli.add_command(psdvariability)


@click.command()
def datasetforbeginners():
	"""
	Copy a set of signals to train using Satis.


	"""
	input_file = pkg_resources.resource_filename(__name__,"Uref_Probes.dat")
	dest_folder = os.path.abspath(__file__)

	fo = open("my_first_dataset.dat","w+")
	fi = open(input_file,"r").readlines()
	for line in fi:
		fo.write(line + '\n')
#	with open(single_signal, 'r') as fin:
#		fin.write(dummy_data)
main_cli.add_command(datasetforbeginners)















