import os
import sys
import subprocess
import argparse
import numpy as np

__all__= ["avbp2global_read_inputs"]

def _read_param(args):
    """ Parse the options (arguments) specified when calling avbp2global.py """

    parser = argparse.ArgumentParser(
        description='Compute the Flame Transfer'
                    ' Function and check its quality with a four steps process',
        epilog='Example: "python avbp2global.py -s 1 -f 290.0"'
               '(for more examples, check the documentation)')

    parser.add_argument(
        '-s',
        '--step',
        default=1,
        type=int,
        choices=list(range(
            1,
            5)),
        required=True,
        dest='step',
        help='Step of the FTF computation process (mandatory).')
    parser.add_argument('-f',
                        '--freq',
                        default=0.0,
                        type=float,
                        required=True,
                        dest='target_frequency',
                        help='Set the target frequency in Hz (mandatory)')
    parser.add_argument('-t',
                        '--tinit',
                        default=0.0,
                        type=float,
                        required=False,
                        dest='transient_time',
                        help='Remove an initial portion of '
                             'the signal in s (optional). '
                             'Using 0.0 deactivates the filter.')
    parser.add_argument('-b',
                        '--bandpass',
                        default=0.0,
                        type=float,
                        required=False,
                        dest='filter_bandwidth',
                        help='Set the frequency bandwidth of '
                             'the bandpass filter (optional). '
                             'Default to two times the frequency resolution.')
    parser.add_argument('-d', '--display',
                        default=True, type=bool,
                        required=False,
                        dest='display_onoff',
                        help='Change the display options (True or False).')
    parser.add_argument('-w',
                        '--write',
                        default=False,
                        type=bool,
                        required=False,
                        dest='savefig',
                        help='Save diagnostic plots as eps files.')
    parser.add_argument('-g',
                        '--debug',
                        default=False,
                        type=bool,
                        required=False,
                        dest='debug',
                        help=('Write intermediate ascii files with diagnostic'
                              ' data.'))

# If no option is specified, return the parser help and quit program
    if len(args) == 1:
        parser.print_help()
        sys.exit(-1)
    else:
        # Otherwise, parse the arguments
        args = parser.parse_args()

# TODO : replace by argparse namespace object
# Return the dictionary of options
    return vars(args)


def _write_choices():
    """ Write default choices file """
    choices = './avbp2global.choices'
    with open(choices, 'w') as template:
        template.write("./avbp_mmm\n")
        template.write("Un1 Vn1 Wn1 ./avbp_local_xxxxxx1\n")
        template.write("Un2 Vn2 Wn2 ./avbp_local_xxxxxx2\n")


def _read_choices():
    """ Read the avbp2global.choices files.
    This file stores the path to the avbp temporal files containing
    the heat release and velocity signals as well as
    the coordinates on which to project the velocities. """

    # Choices file
    choices = './avbp2global.choices'

    # Check if file exists and write out a template if it is not the case
    exists = os.path.exists(choices)
    if not exists:
        print("missing choices files...")
        print("Generating default ....")
        _write_choices()
        sys.exit()

    # Read the choices to provide path for files containing
    # 1) Heat Release (avbp_mmm)
    # 2) Probe velocities (avbp_local_***)

    with open(choices, 'r') as info:

        # 1) Read path for heat release (avbp_mmm)
        #    and extract Q and the total volume
        inputdata = info.readline()
        mmmfile = inputdata.split()[0]

        # 2) Read probes for reference velocity and reference vector
        # coordinates
        reference_probes_vectors = []  # Use a list to preserve order of choices file
        nblocaux = 0

        inputdata = info.readline()
        while inputdata:
            # Check to avoid issues if last line is empty
            if len(inputdata.strip()) > 0:
                data = inputdata.split()
                reference_vector = np.array(list(map(float, data[0:3])))
                probefile = data[3]
                #reference_probes_vectors[probefile] = reference_vector
                reference_probes_vectors.append([probefile, reference_vector])
                nblocaux += 1
            inputdata = info.readline()

        print("--> Found %s probes" % str(nblocaux))

# TODO : return a more explicit output
    return {'mmmfile': mmmfile, 'probefiles': reference_probes_vectors}


def _extract_heat_release(mmmfile, debug):
    """Extract Q(t) and rescale by volume from avbp_mmm"""

    # Check if file exists (should be handled with an exception ?)
    exists = os.path.exists(mmmfile)

    if not exists:
        print("{0} file not found. ".format(mmmfile))
        print("Check its path in the choices file.")
        sys.exit(-1)

    # Execute command and capture error in case of failure
    command = "readbin {0} atime HR_mean Volume".format(mmmfile)
    output = subprocess.check_output(
        command, shell=True, stderr=subprocess.STDOUT)

    # Remove headers and last line
    output = output.split('\n')[3:-1]

    # Convert into a numpy array with three lines (time, HR, volume)
    output = [list(map(float, line.split())) for line in output]
    output = np.transpose(np.array(output))

    # Get volume information (not pertinent for deforming geometries)
    volume = output[2, 0]
    print("\n    Volume : ", volume, "[ m3 ] \n")

    # Remove third column and replace second one by HR*Volume
    output[1] = output[1] * output[2]
    heat_release_signal = output[:2, :]

    if debug:
        np.savetxt('Q.dat', np.transpose(heat_release_signal))

    return heat_release_signal


def _extract_reference_velocity(probefiles, debug):
    ''' Extracts the reference velocities from a set of pairs:
       (probefile, projection_coordinates).
       For each pair, the 2D/3D velocity is read in probefile
       and projected along the coordinates of projection_coordinates.
       '''

    def _normalize_vector(vector):
        ''' Normalize vector '''
        # TODO use np.linalg.norm instead
        norm = np.sqrt(np.sum(vector * vector))
        normalized_vector = vector / norm
        return normalized_vector

    def _read_avbp_probe_file(probefile, reference_vector):
        ''' Read avbp_local_*** file and return projected velocity signal '''

        # Check existence of file
        exists = os.path.exists(probefile)
        assert exists, (
            "{0} file not found. "
            "Check paths and choices").format(probefile)

        # Check if velocity is 3D or 2D
        command = "headbin {0}".format(probefile)
        output = subprocess.check_output(command, shell=True)
        # if DEBUG:
        #    print "       Found vars: \n", output
        if 'w' in output:
            readbin_variables = 'atime u v w'
            ndim = 3
        else:
            readbin_variables = 'atime u v'
            ndim = 2

        # Read binary probefile
        command = "readbin {0} {1}".format(probefile, readbin_variables)
        output = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT)
        # TODO check if stderr=subprocess.STDOUT is correct
        # Is the code below better ?
        #status = subprocess.call(command, shell=True)
        # if status != 0:
        #    raise RuntimeError("Error executing {0}".format(command))
        ###

        # Split output and discard headers and last line
        output = output.split('\n')[4:-1]

        output = [list(map(float, line.split())) for line in output]
        output = np.transpose(np.array(output))

        # Modify second line to store projected velocity
        output[1, :] = output[1, :] * reference_vector[0] + \
            output[2, :] * reference_vector[1]
        if ndim == 3:
            output[1, :] = output[1, :] + output[3, :] * reference_vector[2]

        velocity_signal = output[:2, :]

        return velocity_signal

    print("-> Generating reference velocity data")
    print(" > Treating probe files ...")

    all_velocity_signals = []

    for probedata in probefiles:
        probefile = probedata[0]
        reference_vector = probedata[1]
        # Renormalize reference vector in case
        reference_vector = _normalize_vector(reference_vector)

        print("    -- probe file: {0}".format(probefile))
        print(("       Normal: {0:05.4f}  {1:05.4f}  {2:05.4f}\n"
               ).format(*reference_vector))

        velocity_signal = _read_avbp_probe_file(probefile, reference_vector)
        all_velocity_signals.append(velocity_signal)

    if debug:
        # Extract only velocities
        content = [signal[1, :] for signal in all_velocity_signals]
        # Add time array
        content = [all_velocity_signals[0][0, :]] + content
        np.savetxt(
            'Uref_Probes.dat',
            np.transpose(
                np.array(content)),
            delimiter=' ')

    return np.array(all_velocity_signals)


def avbp2global_read_inputs():
    ''' Read all input parameters for avbp2global tool:
    the arguments, the choices file and the avbp temporal files.'''

    # Obtain the arguments specified when calling the tool
    arguments = sys.argv

    # Obtain the parameters and options
    parameters = _read_param(arguments)
    debug = parameters['debug']

    # Read the avbp2global.choices file and extract heat release
    # and velocities signals.
    avbpfiles = _read_choices()
    raw_heat_release_signal = _extract_heat_release(
        avbpfiles['mmmfile'], debug)
    raw_reference_velocities = _extract_reference_velocity(avbpfiles[
        'probefiles'], debug)

    return parameters, raw_heat_release_signal, raw_reference_velocities
