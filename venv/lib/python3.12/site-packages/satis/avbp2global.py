""" AVBP2GLOBAL tool.

This python tool performs spectral analysis for thermoacoustic applications.
It computes the Flame Transfer Function, with the following procedure:

  1) Read the temporal signals of the global heat release (Q')
     and of the velocities (u') at different locations (a.k.a. "probes").
  2) Compute the Fourier Coefficient of the heat release
     and average velocity signal at the frequency of interest.
  3) Compute the Flame Transfer Function as the ratio between
     the Fourier Coefficient of the Heat Release and
     the Fourier Coefficient of the velocity.

Additionally, the tool provides different quality diagnostics,
gathered into four steps:
Step 1 - Verification of the spectral content of the probes.
Step 2 - Comparison of the Fourier Coefficients of the different velocity probes, in order to identify bad probes.
Step 3 - Check if a permanent regime is reached by plotting running averages / Fourier Transform.
Step 4 - Check the correlation between Q' and u'.

Usage:
  $ python avbp2global_refactored.py [options]

Options:

  -h, --help                    show this help message and exit

  -s STEP, --step=STEP          Step of the procedure.
                                STEP is an integer between 1 and 4.

  -f FREQ, --freq=FREQ          Frequency of interest.
                                FREQ is a float/integer equal to the frequency
                                of interest

  -d DISPLAY, --display=DISPLAY Switch to activate graphical diagnostics or not.
                                Requires matplotlib.
                                DISPLAY is a logical (True/False).

  -b DELTAFF, --bandpass=DELTAF Activate bandpass filtering with a bandwidth of DELTAF.
                                DELTAF is a float/integer equal to the frequency bandwidth.
                                DELTAF= 0.0 deactivates the bandpass filtering.

  -t TINIT, --tinit=TINIT       Truncate the initial TINIT seconds from the signals (to remove transient).
                                TINIT is a float/integer equal to the truncated duration (in seconds).
                                With TINIT = 0.0, the complete signal is used.

  -w WRITE_OPTION, --write=WRITE_OPTION Switch to save intermediate ascii files. WRITE_OPTION is a logical (True/False).

  Options -s and -f are mandatory when calling the command.

Examples:

  $ python avbp2global_refactored.py -s 1 -f 290.0
  $ python avbp2global_refactored.py --step=1 --freq=290.0
  $ python avbp2global_refactored.py -s 1 -f 290.0, -d True -b 10.0, -t 0.001
  $ python avbp2global_refactored.py --step=1 --freq=290.0 --display=True

Inputs :

  - avbp2global.choices (generated automatically when the tool is called for
    the first time)
    Format:
    Line 1: Path to avbp_mmm file
    Line 2 and following : Coordinates of the reference vector +
    Path to the avbp_local_*** file

  - avbp_mmm (from an AVBP computation)

  - avbp_local_*** (from an AVBP computation)

Outputs :

  If display option is activated, a matplotlib figure containing diagnostics
  is opened.

  In all cases, the FTF parameters (gain and delay) are displayed in the
  terminal.
"""

import numpy as np

from satis.avbp2global_compute_ftf import get_flame_transfer_function
from satis.avbp2global_read_inputs import avbp2global_read_inputs

__all__= ["display_header"]


def display_header():
    """ Display tool welcome message """
    print("-----------------------------------------------------------------")
    print("                 WELCOME TO AVBP2GLOBAL                          ")
    print("-----------------------------------------------------------------")
    print(" ")


# ==============================================================================
if __name__ == "__main__":

    # Display tool welcome message
    display_header()

    # Read input parameters
    PARAMETERS, HEAT_RELEASE_DATA, \
        VELOCITY_DATA = avbp2global_read_inputs()

    DISPLAY = PARAMETERS['display_onoff']

    if DISPLAY:
        import libavbp2global.avbp2global_diagnostics as a2g_diag
        a2g_diag.diagnostic(HEAT_RELEASE_DATA, VELOCITY_DATA, PARAMETERS)
    else:
        print ("-> The display of data is not activated. "
               "Change the --display option (requires Matplotlib).")

    # Get time, heat release and velocities arrays
    TIME = HEAT_RELEASE_DATA[0, :]
    HEAT_RELEASE = HEAT_RELEASE_DATA[1, :]
    AVERAGE_VELOCITY = np.average(VELOCITY_DATA[:, 1, :], axis=0)

    # Compute and display in terminal the flame transfer function.
    # Replace _, _, _ by gain, phase and delay variables if desired.
    _, _, _ = get_flame_transfer_function(
        TIME, HEAT_RELEASE, AVERAGE_VELOCITY, PARAMETERS, show=True)
