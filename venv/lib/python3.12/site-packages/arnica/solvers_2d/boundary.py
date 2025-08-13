"""
Contains class and functions necessary to handle boundary conditions
"""

import numpy as np
import scipy.sparse as spsp
from arnica.solvers_2d.core_fd import set_row_csr


class Boundary2d:
    """
    *Class containing Boundary Condition information*
    """

    def __init__(self, boundary_param):
        """
        Constructor of boundary class
        """
        self.boundary_params = boundary_param

        print('\n\t\t Boundary2D:\t _set_periodicity()')
        self._set_periodicity()

    def _set_periodicity(self):
        """
        Set the periodicity to North/South if periodic and check for the
        consistency of the definition of the BC
        """
        
        bc_n = self.boundary_params["North"]["type"]
        bc_w = self.boundary_params["West"]["type"]
        bc_s = self.boundary_params["South"]["type"]
        bc_e = self.boundary_params["East"]["type"]

        self.periodic_we = False
        self.periodic_ns = False

        if bc_n == "Periodic" or bc_s == "Periodic":
            if bc_n == bc_s:
                self.periodic_ns = True
            else:
                raise IOError(
                    'Both "North" and "South" Boundary Conditions need to \
                    be defined as "Periodic" for the periodic treatment to \
                    be applied')

        if bc_w == "Periodic" or bc_e == "Periodic":
            if bc_w == bc_e:
                self.periodic_we = True
            else:
                raise IOError(
                    'Both "West" and "East" Boundary Conditions need to \
                    be defined as "Periodic" for the periodic treatment to \
                    be applied')


def apply_bc(lhs, rhs, metric, boundaries):
    """
    Give the altered version of LHS matrix and RHS vector to apply boundary
    conditions

    Parameters
    ----------
    lhs: left-hand side matrix (A in AX=B)
    rhs: right-hand side matrix (B in AX=B)
    metric: an instance of class Metrics2d containing gradient operators
    boundaries: dictionary with boundary data

    Returns
    -------
    lhs: modified left-hand side matrix
    rhs: modified right-hand side matrix
    """

    for boundary_name, bc_dict in boundaries.items():

        if bc_dict["type"] == 'Robin':
            a_value, b_value, c_value = bc_dict["bc_values"]
            nodes = metric.bnd_nodes[boundary_name]
            grad_n = metric.grad_n_bc
            bc_robin(lhs, rhs, nodes, grad_n, a_value, b_value, c_value)

        elif bc_dict["type"] == 'Neumann':
            # Todo: code Neumann from Robin BC
            pass
        elif bc_dict["type"] == 'Dirichlet':
            dirichlet_value = bc_dict["bc_values"]
            nodes = metric.bnd_nodes[boundary_name]
            grad_n = metric.grad_n_bc
            bc_dirichlet(lhs, rhs, nodes, grad_n, dirichlet_value)
        elif bc_dict["type"] == 'Periodic':
            pass
        else:
            raise NotImplementedError('The only available boundary conditions \
                                      are "Robin", "Neumann", "Dirichlet" and \
                                      "Periodic".')

    return lhs, rhs


# Todo : passer en csr
def bc_robin_csr(matrix_lhs, matrix_rhs, positions, grad_n_bc, a_val,
                 b_val, c_val):
    """
    Function to enforce the Boundary Condition in the Robin formalism:

                    a * (df / dn) + b * f + c = O

    Parameters
    ----------
    matrix_lhs: [CSR] Left Hand Side square matrix
                        (shape (size_i_w * size_j_ns)^2)
    matrix_rhs: [ndarray] Right Hand Side vector
                        (shape (size_i_w * size_j_ns))
    positions: indices of lines corresponding to the patch being processed
    grad_n_bc: [CSR] normal gradient to boundary nodes
    a_val: value of the "a" Robin parameter
    b_val: value of the "b" Robin parameter
    c_val: value of the "a" Robin parameter

    Returns
    -------
    Altered left-hand side matrix and right-hand side vector
    """

    for row in positions:
        # Gradient normal to the boundary if necessary
        if grad_n_bc.count_nonzero():
            new_row = a_val * grad_n_bc.getrow(row).toarray().ravel()
            set_row_csr(matrix_lhs, row, new_row)

        # # Term "b" of robin BC
        diag_bc = np.zeros(matrix_rhs.shape)
        diag_bc[row] = b_val
        mat_bc_b_csr = spsp.csr_matrix(np.diag(diag_bc))
        matrix_lhs = matrix_lhs + mat_bc_b_csr

    # Right hand side for inhomogeneous Dirichlet / Neumann / Robin BCs
    matrix_rhs[positions] = - c_val

    return matrix_lhs, matrix_rhs


def bc_robin(matrix_lhs, matrix_rhs, positions, grad_n_bc, a_val,
             b_val, c_val):
    """
    Function to enforce the Boundary Condition in the Robin formalism:

                    a * (df / dn) + b * f + c = O

    Parameters
    ----------
    matrix_lhs: Left Hand Side square matrix (shape (size_i_w * size_j_ns)^2)
    matrix_rhs: Right Hand Side vector (shape (size_i_w * size_j_ns))
    positions: indices of lines corresponding to the patch being processed
    grad_n_bc: normal gradient to boundary nodes
    a_val: value of the "a" Robin parameter
    b_val: value of the "b" Robin parameter
    c_val: value of the "a" Robin parameter

    Returns
    -------
    Altered left-hand side matrix and right-hand side vector
    """

    for row in positions:
        # Gradient normal to the boundary
        if grad_n_bc.count_nonzero():
            matrix_lhs[row, :] = a_val * grad_n_bc[row, :]

        # # Term "b" of robin BC
        matrix_lhs[row, row] += b_val

    # Right hand side for inhomogeneous Dirichlet / Neumann / Robin BCs
    matrix_rhs[positions] = - c_val

    return matrix_lhs, matrix_rhs


def bc_neumann(matrix_lhs, matrix_rhs, positions, grad_n_bc,
               target_gradient):
    """
    Function to enforce the Neumann Boundary Condition in the Robin formalism:

                    a * (df / dn) + b * f + c = O

    Parameters
    ----------
    grad_n_bc
    target_gradient
    matrix_lhs : Left Hand Side square matrix (shape (size_i_w * size_j_ns)^2)
    matrix_rhs : Right Hand Side vector (shape (size_i_w * size_j_ns))
    positions : indices of lines corresponding to the patch being processed

    Returns
    -------
    Altered left-hand side matrix and right-hand side vector
    """
    lhs_out, rhs_out = bc_robin(matrix_lhs, matrix_rhs, positions,
                                grad_n_bc, 1., 0., -target_gradient)

    return lhs_out, rhs_out


def bc_dirichlet(matrix_lhs, matrix_rhs, positions, grad_n_bc, target_value):
    """
    Function to enforce Dirichlet Boundary Condition in the Robin formalism:

                    a * (df / dn) + b * f + c = O

    Parameters
    ----------
    target_value
    matrix_lhs : Left Hand Side square matrix (shape (size_i_w * size_j_ns)^2)
    matrix_rhs : Right Hand Side vector (shape (size_i_w * size_j_ns))
    positions : indices of lines corresponding to the patch being processed

    Returns
    -------
    Altered left-hand side matrix and right-hand side vector
    """
    print("Dirichlet bc target -->", target_value)

    # lhs_out, rhs_out = bc_robin_csr(matrix_lhs, matrix_rhs, positions,
    #                                 grad_n_bc, 0., 1., -target_value)
    lhs_out, rhs_out = bc_robin(matrix_lhs, matrix_rhs, positions,
                                grad_n_bc, 0., 1., -target_value)

    return lhs_out, rhs_out
