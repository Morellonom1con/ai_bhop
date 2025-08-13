""" script to run scenarii for pyhip"""
from pyhip.scenarii import complete_setup, interpolation_setup, adaptation_setup, basic_setup
import yaml
from nob import Nob


def basic_scenario():
    """basic scenario launcher """
    with open('basic.yml', 'r') as yml_file:
        nob_in = Nob(yaml.load(yml_file, Loader=yaml.FullLoader))

    _ = basic_setup(nob_in, nob_in.fallback[:])

def complete_scenario():
    """complete scenario launcher """
    with open('complete.yml', 'r') as yml_file:
        nob_in = Nob(yaml.load(yml_file, Loader=yaml.FullLoader))

    _ = complete_setup(nob_in, nob_in.fallback[:], nob_in.checklevel[:])

def interpolation_scenario():
    """interpolation scenario launcher """
    with open('interpolation.yml', 'r') as yml_file:
        nob_in = Nob(yaml.load(yml_file, Loader=yaml.FullLoader))

    _ = interpolation_setup(nob_in, nob_in.fallback[:])

def adapt_scenario():
    """adapt scenario launcher """
    with open('adapt.yml', 'r') as yml_file:
        nob_in = Nob(yaml.load(yml_file, Loader=yaml.FullLoader))

    _ = adaptation_setup(nob_in, nob_in.fallback[:])

if __name__ == "__main__":
    """uncomment the ones you wish to use, repolace the in correct order if needed"""
    #basic_scenario()
    complete_scenario()
    #interpolation_scenario()
    #adapt_scenario()

