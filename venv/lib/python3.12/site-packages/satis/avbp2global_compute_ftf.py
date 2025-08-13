import numpy as np
from satis.satis_library import get_coeff_fourier

__all__= ["get_flame_transfer_function"]


def get_flame_transfer_function(
        time, heat_release, velocity, parameters, show=False):
    ''' Compute the gain, delay and phase of the Flame Transfer Function.
         Display the results in the terminal. '''

    
    # Get useful parameters
    target_frequency = parameters['target_frequency']
    transient_time = parameters['transient_time']

    # Compute Fourier coefficients at f=freq

    fourier_heat_release = get_coeff_fourier(
        time, heat_release, target_frequency, transient_time)
    fourier_velocity = get_coeff_fourier(
        time, velocity, target_frequency, transient_time)

    # Compute the gain and delay of the Flame Transfer Function
    flame_transfer_function = fourier_heat_release / fourier_velocity
    gain = np.abs(flame_transfer_function)
    # Minus sign in phase to be consistent with exp(-iwt) convention in AVSP.
    phase = -np.angle(flame_transfer_function) % (2 * np.pi)
    delay = phase / (2 * np.pi * target_frequency)

    mean_heat_release = np.mean(heat_release)
    mean_velocity = np.mean(velocity)
    
    if show:

        print('')
        print('--- FTF results ---')
        print("-----------------------------------------------------------------")
        print(" Umean = {0:04.2f} [ m/s ], Qmean = {1:04.2f} [ J/s ]".format(
            mean_velocity, mean_heat_release))
        print(" ")
        print(" Gain = {0:08.6f} [ J/m ], Delay = {1:08.6f} [ ms ]".format(
            gain, delay * 1.0e3))
        print(" ")
        print(" Adimentionalised equivalent :")
        print(" ")
        print(" Gain = {0:08.6f}, Delay = {1:08.6f}".format(
            gain * mean_velocity / mean_heat_release, delay * target_frequency))
        print("-----------------------------------------------------------------")
        print(" ")   

    return gain, delay, phase
