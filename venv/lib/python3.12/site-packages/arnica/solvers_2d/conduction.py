""" Module to test the heat transfer
resolution on grids curvilinear 2D grids """

import scipy.sparse as scp
# import scipy.sparse.linalg as linalg
import numpy as np
import arnica.utils.mesh_tool as msh
from arnica.solvers_2d.core_fd import Metrics2d
from arnica.solvers_2d import boundary as bndy
from arnica.utils.nparray2xmf import NpArray2Xmf


def compute_constants(params):
    """ compute_constants parameters

    Parameters :
    ------------
    params : dict of setup parameters


    Returns :
    ---------
    const : dict of constant parameters
    """

    const = {}
    const["h_ratio"] = params["h_cold"] / params["h_hot"]
    const["biot"] = params["h_hot"] * params["width"] / params["lambda"]
    const["t_ref"] = (params["t_cold"]
                      + ((params["t_hot"] - params["t_cold"])
                         * (1. / (1. + const["h_ratio"]))))
    const["dx"] = params["typ_size"] / params["res"]

    const["lam_ov_rhocp"] = params["lambda"] / (params["rho"] * params["cp"])

    const["dt"] = params["fourier"] * const["dx"] ** 2 / const["lam_ov_rhocp"]
    const["total_time"] = params["typ_size"] ** 2 / const["lam_ov_rhocp"]
    const["exp_iterations"] = const["total_time"] / const["dt"]

    for key in const:
        print(key
              + ":"
              + str(const[key]))

    return const


def slope_function(coor, min_val, max_val):
    """
    create a slope x test function, with analytically known derivatives

    Parameters :
    ------------
    coor : numpy array (n,m) , coordinates
    min_val : minimum value
    max_val ; maximum value

    Returns :
    ---------
    slop : numpy array (n,m) , slope test function

    """

    min_c = coor.min()
    max_c = coor.max()

    out = np.ones_like(coor) * min_val
    out += (coor - min_c) / (max_c - min_c) * (max_val - min_val)

    return out


def build_heat_equation(metric):
    """
    Create the LHS matrix corresponding to heat equation
    Parameters
    ----------
    metric : an instance of the class Metrics2d containing the gradient operators

    Returns
    -------
    LHS matrix
    RHS matrix
    """

    # Left Hand Side
    grad_x = metric.grad_x_csr
    grad_y = metric.grad_y_csr
    laplacian = grad_x.dot(grad_x) + grad_y.dot(grad_y)

    lhs = laplacian

    # Right-Hand Side
    rhs = np.zeros(metric.shp1d)

    return lhs, rhs


def set_bc_heat_transfer(bc_params):
    """
    Fill the generic bc_values from wall_temperature.
    (bc_dirichlet function is generic, we have to tell the solver that the
    value that we want to impose is the wall temperature here)

    Parameters
    ----------
    bc_params: dictionary of boundary parameters

    Returns
    -------
    bc_params: updated dictionary of boundary parameters
    """

    for _, bc_dict in bc_params.items():

        if bc_dict["type"] == 'Dirichlet':
            bc_dict["bc_values"] = bc_dict["Values"]["wall_temperature"]

    return bc_params


def main_2d_condution(params):
    """ Set up main 2D heat transfer resolution

    Parameters :
    ------------
    params : dict of setup parameters

    """

    print("\n\tmain ---> Create mesh")
    x_coor, y_coor = msh.get_mesh(params["mesh"])

    if params['mesh']["dilate_grid"]:
        x_coor, y_coor, _ = msh.dilate_center(x_coor, y_coor)

    print("\n\tmain ---> Define Boundaries")
    bnd = bndy.Boundary2d(params["boundaries"])

    print("\n\tmain ---> Compute Metrics")
    metric = Metrics2d(x_coor,
                       y_coor,
                       periodic_ns=bnd.periodic_ns,
                       periodic_we=bnd.periodic_we)

    print("\n\tmain --> Build LHS and RHS")
    lhs_csr, rhs_csr = build_heat_equation(metric)

    temp_east = params["boundaries"]["East"]["Values"]["wall_temperature"]
    temp_west = params["boundaries"]["West"]["Values"]["wall_temperature"]
    temp_final = slope_function(x_coor, temp_west, temp_east)

    temp0_1d = np.ones(metric.shp1d) * temp_east
    # temp0_2d = temp0_1d.reshape(metric.shp2d)

    if params["solver"]["method"] == "iterative":
        # mat_bc_csr, rhs = apply_bc_csr(metric.lapl, metric.size_i,
        #                                metric.size_j,
        #                                params["t_cold"], params["t_hot"])

        params["boundaries"] = set_bc_heat_transfer(params["boundaries"])

        lhs_csr_bc, rhs_csr_bc = bndy.apply_bc(lhs_csr, rhs_csr, metric,
                                               params["boundaries"])

        temp1d, info = scp.linalg.bicgstab(lhs_csr_bc, rhs_csr_bc, x0=temp0_1d)

        print(">> Result bicg ", info)
        temp = temp1d.reshape(metric.shp2d)

    # if params["method"] == "direct":
    #     const = compute_constants(params)
    #     temp = explicit_solve(
    #         temp0_2d,
    #         metric,
    #         params,
    #         const,
    #         temp_final)

    def gap(temp):
        """ return the maximum absolute error """
        return np.abs(temp_final - temp).max()

    print("Max error:", gap(temp))

    outh5 = NpArray2Xmf("out.h5")
    outh5.create_grid(x_coor, y_coor, np.zeros(metric.shp2d))
    outh5.add_field(temp, "temp")
    outh5.add_field(temp_final, "temp_analytical")
    outh5.add_field(temp_final - temp, "temp_error")
    outh5.dump()

def example_params():
    """Example input"""
    # Properties
    temp_west = 600.
    h_west = 3000.

    temp_east = 1500.
    h_east = 1500.

    # Mesh definition
    mesh_rect = {"kind": 'rect',
                 "x_max": 1.,
                 "y_max": 1.,
                 "x_res": 51.,
                 "y_res": 50.,
                 "dilate_grid": False}

    # Boundary conditions
    # Default
    boundary = {"North": {"type": "Periodic", 'Values': {}},
                "West": {"type": "Periodic", 'Values': {}},
                "South": {"type": "Periodic", 'Values': {}},
                "East": {"type": "Periodic", 'Values': {}}}

    boundary["West"]["type"] = "Dirichlet"
    boundary["West"]["Values"] = {"wall_temperature": temp_west,
                                  "heat_transfer_coeff": h_west}

    boundary["East"]["type"] = "Dirichlet"
    boundary["East"]["Values"] = {"wall_temperature": temp_east,
                                  "heat_transfer_coeff": h_east}

    field = {}
    field['rho'] = 7900.
    field['cp'] = 435.
    field['lambda'] = 11.4

    # Solver options
    solver = {"method": "iterative",
              # "method": "direct",
              "tol": 1.0,
              "fourier": 0.8}

    # Create parameter dictionary
    params = {}
    params["boundaries"] = boundary
    params["mesh"] = mesh_rect
    params["field"] = field
    params["solver"] = solver
    return params

if __name__ == "__main__":
    PARAMS = example_params()
    main_2d_condution(PARAMS)
