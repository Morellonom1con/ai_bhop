""" module to compute wall equilibrium """
import numpy as np

__all__ = ["compute_equilibrium"]


def compute_equilibrium(
        hot_t_ad,
        cold_t_ad,
        hot_h,
        cold_h,
        metal,
        ceram,
        ep_metal,
        ep_ceram):
    """ 
    compute the wall equilibrium
    -----------------------------

    ::

                - - - - - - - >
                HOT SIDE  hot_h, hot_t_ad

             ^ phi
            _|___________________  t_ceram_hot
             |       Layer 2
            _|___________________ t_eq
             |       Layer 1
            _|___________________ t_metal_hot
             |
                COLD SIDE cold_h, cold_t_ad
                - - - - >
    """

    t_eq = np.copy(cold_t_ad)
    t_eq_m1 = np.copy(cold_t_ad)

    def estimate_phi_and_temps(t_ceram, t_metal):
        """ return phi and temperatures """
        r_th_ceram = ceram.thermal_resistance(ep_ceram, t_ceram)
        r_th_metal = metal.thermal_resistance(ep_metal, t_metal)

        phi = (hot_t_ad - cold_t_ad) / (1.0 / hot_h +
                                        r_th_ceram +
                                        r_th_metal +
                                        1.0 / cold_h)

        t_ceram_hot = hot_t_ad - phi / hot_h
        t_ceram_metal = t_ceram_hot - phi * r_th_ceram
        t_metal_cold = t_ceram_metal - phi * r_th_metal

        return t_ceram_hot, t_ceram_metal, t_metal_cold

    t_ceram_hot, t_eq, t_metal_cold = estimate_phi_and_temps(t_ceram=t_eq,
                                                             t_metal=t_eq)

    # Stop convergence at 1K
    ncount = 0
    cvgce = 1000.
    while cvgce > 1.0:
        ncount += 1
        t_eq_m1 = t_eq
        t_ceram_hot, t_eq, t_metal_cold = estimate_phi_and_temps(
            t_ceram = 0.5 * (t_ceram_hot + t_eq),
            t_metal = 0.5 * (t_metal_cold + t_eq))
        cvgce =  np.max(np.abs(t_eq - t_eq_m1)) 

        if ncount > 30:
            msgerr = "Compute_equilibrium loop did not conveg."
            msgerr = "Breaking with a deltaT of  " + str(cvgce) + "K"
            raise RuntimeWarning(msgerr)

    return t_ceram_hot, t_eq, t_metal_cold
