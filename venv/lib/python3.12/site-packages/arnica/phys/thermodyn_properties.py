"""Module for computing thermodynamic properties
"""

import numpy as np


__all__ = [
    "thermal_constants",
    "h_kader",
    "lambda_cp_visco_fluid",
    "viscosity_sutherland",
    "fluid_cp"
]

def thermal_constants():
    """Generate a dictionnary of thermal constants

    Returns:
        TYPE: Description
    """
    consts = dict()
    consts['prandtl'] = 0.71
    consts['prandtl_turb'] = 0.86
    consts['vkarman'] = 0.41
    consts['van_driest'] = 5.5
    consts['beta_kader'] = ((3.85 * consts['prandtl']**(1.0/3.0) - 1.3)**2
                            + 2.12 * np.log(consts['prandtl']))
    consts['k_kader'] = ((consts['beta_kader']
                          - consts['prandtl_turb'] * consts['van_driest']
                          + consts['prandtl_turb'] / consts['vkarman'] - 2.12)
                         - (consts['prandtl_turb'] / consts['vkarman'] - 2.12)
                         * (200.0*np.log(200.0) - 100*np.log(100.0))
                         / (200.0 - 100.0))
    return consts


def _estimate_log_region(y_plus_guess, u_2, y_rho_mu_wall):
    """evaluate u tau in log region

    Returns:
        TYPE: Description

    Args:
        y_plus_guess (TYPE): Description
        u_2 (TYPE): Description
        y_rho_mu_wall (TYPE): Description
    """
    u_tau_log = 0
    epsilon = 1.0
    y_plus_log = y_plus_guess
    consts = thermal_constants()

    def _loop_u_tau(u_tau_log, y_plus_log):
        """recursive loop to evaluate u_tau

        Args:
            u_tau_log (TYPE): Description
            y_plus_log (TYPE): Description

        Returns:
            TYPE: Description

        """
        utau_old = u_tau_log
        u_tau_log = u_2 / (1 / consts['vkarman'] * np.log(y_plus_log)
                           + consts['van_driest'])
        y_plus_log = u_tau_log * y_rho_mu_wall
        epsilon = np.abs(u_tau_log - utau_old)
        return u_tau_log, y_plus_log, epsilon

    while np.min(epsilon) >= 1.e-5:
        (u_tau_log, y_plus_log, epsilon) = _loop_u_tau(u_tau_log, y_plus_log)
    return u_tau_log, y_plus_log

def _estimate_t_tau(u_tau, y_plus, u_2, t_wall, t_2):
    """Compute T_tau

    Args:
        u_tau (TYPE): Description
        y_plus (TYPE): Description
        u_2 (TYPE): Description
        t_wall (TYPE): Description
        t_2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    consts = thermal_constants()
    kader_g = (0.01 * (consts['prandtl'] * y_plus['all'])**4)
    kader_g /= (1.0 + 5.0 * consts['prandtl']**3 * y_plus['all'])
    t_tau_cwm = (
        (t_wall - t_2)
        / ((consts['prandtl'] * y_plus['all'] * np.exp(-kader_g))
           + (consts['prandtl_turb']*u_2/u_tau['cwm'] +  consts['k_kader'])
           * np.exp(-1.0/kader_g)))
    return t_tau_cwm

def h_kader(t_wall, rho_wall, y_wall, u_2, t_2, temp_adiab):
    """compute h at the wall as in kader
        names taken equalt to  loglaw_cwm.f90 AVBP

    Args:
        t_wall (TYPE): Description
        rho_wall (TYPE): Description
        y_wall (TYPE): Description
        u_2 (TYPE): Description
        t_2 (TYPE): Description
        temp_adiab (TYPE): Description

    Returns:
        TYPE: Description
    """
    _, heat_cp, mu_wall = lambda_cp_visco_fluid(t_2)

    y_plus = dict()
    y_rho_mu_wall = y_wall * rho_wall / mu_wall
    y_plus['guess'] = 0.157 * (u_2 * y_rho_mu_wall)**(7.0/8.0)
    y_plus['guess'] = np.maximum(y_plus['guess'], 1.0e-3)

    u_tau = dict()
    # Linear part
    u_tau['lin'] = np.sqrt(u_2 / y_rho_mu_wall)

    #Log part
    u_tau['log'], y_plus['log'] = _estimate_log_region(y_plus['guess'], u_2,
                                                       y_rho_mu_wall)

    # fusion of the two
    u_tau['cwm'] = np.where(y_plus['guess'] < 11.25, u_tau['lin'], u_tau['log'])
    y_plus['all'] = u_tau['cwm'] * y_rho_mu_wall

    t_tau_cwm = _estimate_t_tau(u_tau, y_plus, u_2, t_wall, t_2)
    h_wall = -(rho_wall * heat_cp * u_tau['cwm'] * t_tau_cwm)
    h_wall /= (temp_adiab - t_wall)

    return h_wall


def lambda_cp_visco_fluid(temperature):
    """compute Fluid properties lambda , cp, visco

    Args:
        temperature (TYPE): Description

    Returns:
        TYPE: Description
    """
    consts = thermal_constants()
    heat_cp = fluid_cp(temperature)
    mu_wall = viscosity_sutherland(temperature)
    lam = heat_cp / consts['prandtl'] * mu_wall

    return lam, heat_cp, mu_wall

def viscosity_sutherland(temp):
    """compute visocity as in sutherland

    Args:
        temp (TYPE): Description

    Returns:
        TYPE: Description

    """
    alpha = 1.716e-5
    beta = 1.5
    t_0 = 273.15
    t_1 = 110.4
    mu_wall = alpha * (temp / t_0)**beta * (t_0 + t_1) / (temp + t_1)
    return mu_wall

def fluid_cp(temp, clipping=False):
    """compute cp of fluid

    Args:
        temp (TYPE): Description
        clipping (bool, optional): Description

    Returns:
        TYPE: Description

    """
    t_clip = np.array(temp)
    if clipping:
        t_clip = np.clip(t_clip, 300.0, 2500.0)
    coeffs = [513.46, 1.6837, -1.9275e-3, 1.2773e-6, -4.2773e-10, 5.6735e-14]
    heat_cp = (
        coeffs[0] * np.ones_like(t_clip)
        + coeffs[1] * t_clip
        + coeffs[2] * np.power(t_clip, 2)
        + coeffs[3] * np.power(t_clip, 3)
        + coeffs[4] * np.power(t_clip, 4)
        + coeffs[5] * np.power(t_clip, 5)
    )
    return heat_cp

def h_stanton(stanton, rhou, c_p):
    """return the H correlation from a stanton number

    Ok really a no brainer"""

    return stanton*rhou*c_p


def stanton_colburn(reynolds, prandtl):
    """return the stanton from a colburn correlation
    
    https://en.wikipedia.org/wiki/Stanton_number#Correlations_using_Reynolds-Colburn_analogy
    Lienhard, John H. (2011). A Heat Transfer Textbook. Courier Corporation. p. 313. ISBN 978-0-486-47931-6.
    """

    if reynolds.min() < 1./0.06 :
        print("Warning : Reynolds < 16.6, this Colburn correlation does not apply, reynolds clip")
        reynolds=np.clip(reynolds, 100., None)

    c_f = 0.455 / (np.log(0.06 * reynolds))**2
    st_ = c_f*0.5 / (  
        1.+ 12.8 * (np.power(prandtl, 0.68) - 1.) * np.sqrt(c_f*0.5)
    )

    return st_


def h_colburn(rhou, temperature, char_len):
    """return the H correlation from a Coburn correlation
    
    rhou :  momentum [Kg/m2 /s]
    temperature :  temperature of the fluid foir constants correls
    char_len :  characteristic lenght for Reynolds [m]

    returns : 
    =========
    h_col : colburn h
    
    """
    _, _, _, h_coeff = h_colburn_detailed(rhou, temperature, char_len)

    return h_coeff

def h_colburn_detailed(rhou, temperature, char_len):
    """return the H correlation from a Coburn correlation

    rhou :  momentum [Kg/m2 /s]
    temperature :  temperature of the fluid foir constants correls
    char_len :  characteristic lenght for Reynolds [m]

    returns : 
    =========
    h_col : colburn h
    
    """
    lam, heat_cp, mu_wall = lambda_cp_visco_fluid(temperature)
 
    prandtl = heat_cp * mu_wall / lam
    reynolds = rhou * char_len / mu_wall
    stanton = stanton_colburn(reynolds,prandtl)
    h_coeff = h_stanton(stanton,rhou, heat_cp)
    return prandtl, reynolds, stanton, h_coeff