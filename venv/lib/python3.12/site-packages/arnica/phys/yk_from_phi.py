"""
This script calculate mass_fraction of species from a Phi
"""

__all__ = ["yk_from_phi",
           "phi_from_far"]

def phi_from_far(far, c_x, h_y):
    """
    *Return phi coefficient with the fuel air ratior coeff + fuel composition.*

    :param far: the air-fuel ratio
    :type far: float
    :param c_x: stoechio coeff of Carbone
    :type c_x: float
    :param h_y: stoechio coeff of hydrogene
    :type h_y: float
    """
    mass_molar = {
        "C": 0.0120107,
        "H": 0.00100797,
        "O2": 0.0319988,
        "N2": 0.0280134
    }
    mass_mol_fuel = (c_x * mass_molar["C"] + h_y * mass_molar["H"])
    coeff_o2 = c_x + (h_y / 4)
    y_o2_fuel_sto = (coeff_o2 * mass_molar["O2"]) / mass_mol_fuel
    phi = far / y_o2_fuel_sto

    return phi

def yk_from_phi(phi, c_x, h_y):
    """ 
    *Return the mass fraction of elements from a fuel aspect ratio and stoechio element coeff.*

    :param phi: the air-fuel aspect ratio
    :type phi: float
    :param c_x: stoechio coeff of Carbone
    :type c_x: float
    :param h_y: stoechio coeff of hydrogene
    :type h_y: float

    """
    y_k = dict()
    mass_molar = {
        "C": 0.0120107,
        "H": 0.00100797,
        "O2": 0.0319988,
        "N2": 0.0280134
    }

    mass_mol_fuel = (c_x * mass_molar["C"] + h_y * mass_molar["H"])
    coeff_o2 = c_x + (h_y / 4)
    y_o2_fuel_sto = (coeff_o2 * mass_molar["O2"]) / mass_mol_fuel

    if phi == 0.:
        y_k["fuel"] = 0.
    else:
        y_k["fuel"] = (1. / (1. + (
            1.
            + 3.76 * (mass_molar["N2"] / mass_molar["O2"]))
            * (y_o2_fuel_sto / phi))
        )
    y_air = 1 - y_k["fuel"]
    y_k["N2"] = y_air / 1.303794
    y_k["O2"] = 0.303794 * y_k["N2"]

    return y_k
