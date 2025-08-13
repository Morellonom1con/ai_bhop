"""
This module is contains the core of Finite Differences 2D solvers.

The class metrics:
    1) Builds the jacobian to handle curvilinear meshes.
    2) Computes the gradients operators
    3) Computes the boundary-normal-gradient operator
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as scp

from arnica.utils.nparray2xmf import NpArray2Xmf

WRITE_BNDY = False


# Todo : think about axi-periodic treatments
class Metrics2d:
    """
    Compute operators onto a curvilinear 2D mesh
    """

    def __init__(self, x_coor, y_coor, periodic_ns=False, periodic_we=False):
        """
        Metric2d constructor

        Parameters
        ----------
        x_coor: x_coordinates as a numpy meshgrid
        y_coor: y_coordinates as a numpy meshgrid
        periodic_ns: Boolean, True if periodic North-South BC
        periodic_we: Boolean, True if periodic West-East BC
        """
        self.x_coor = x_coor
        self.y_coor = y_coor

        self.size_we, self.size_ns = x_coor.shape
        self.size = self.size_we * self.size_ns

        self.shp2d = x_coor.shape
        self.shp1d = x_coor.ravel().shape

        self.periodic_ns = periodic_ns
        self.periodic_we = periodic_we

        self.grad_x_slow = None
        self.grad_y_slow = None

        print('\n\t\t Metric:\t _compute_metric()')
        self._compute_metric()

        print("\n\t\t Metric:\t  _compute_matrices_csr()")
        self._compute_matrices_csr()

        print("\n\t\t Metric:\t  _get_bc_iterators()")
        self._get_bc_iterators()

        print("\n\t\t Metric:\t  _compute_normals()")
        self._compute_normals()

    def _compute_metric(self):
        """
        Compute the metric elements
        """
        diff_x_ns = self.diff_ns_2(self.x_coor)
        diff_x_we = self.diff_we_2(self.x_coor)
        diff_y_ns = self.diff_ns_2(self.y_coor)
        diff_y_we = self.diff_we_2(self.y_coor)
        inv_det_j = np.reciprocal(
            diff_x_ns * diff_y_we - diff_x_we * diff_y_ns)

        self.jacob_x_ns = inv_det_j * diff_x_ns
        self.jacob_x_we = inv_det_j * diff_x_we
        self.jacob_y_ns = inv_det_j * diff_y_ns
        self.jacob_y_we = inv_det_j * diff_y_we

    def check_compute_matrices(self):
        """
        compute the matrix form of operators
        DO NOT REMOVE! --> USED FOR TESTING CSR GRADIENTS
        """

        def ijc(i_idx, j_idx):
            """ Give indexes in 1D version """
            return j_idx + self.size_ns * i_idx

        def iters(k, size_k, perio):
            """Provide iterators, taking boundaries into account """

            cfk = 0.5
            km1 = k - 1
            kp1 = k + 1
            if k == 0:
                if perio:
                    km1 = size_k - 1
                else:
                    cfk = 1.
                    km1 = 0

            if k == size_k - 1:
                if perio:
                    kp1 = 0
                else:
                    cfk = 1.
                    kp1 = size_k - 1

            return km1, kp1, cfk

        mat_gx = np.zeros((self.size, self.size))
        mat_gy = np.zeros((self.size, self.size))

        for i_we in range(self.size_we):
            im1, ip1, cfi = iters(i_we, self.size_we, self.periodic_ns)
            for j_ns in range(self.size_ns):
                jm1, jp1, cfj = iters(j_ns, self.size_ns, self.periodic_we)

                jcb_y_we = self.jacob_y_we[i_we, j_ns]
                jcb_y_ns = self.jacob_y_ns[i_we, j_ns]
                jcb_x_we = self.jacob_x_we[i_we, j_ns]
                jcb_x_ns = self.jacob_x_ns[i_we, j_ns]

                mat_gx[ijc(i_we, j_ns), ijc(i_we, jp1)] += cfj * jcb_y_we
                mat_gx[ijc(i_we, j_ns), ijc(i_we, jm1)] -= cfj * jcb_y_we
                mat_gx[ijc(i_we, j_ns), ijc(ip1, j_ns)] -= cfi * jcb_y_ns
                mat_gx[ijc(i_we, j_ns), ijc(im1, j_ns)] += cfi * jcb_y_ns

                mat_gy[ijc(i_we, j_ns), ijc(i_we, jp1)] -= cfj * jcb_x_we
                mat_gy[ijc(i_we, j_ns), ijc(i_we, jm1)] += cfj * jcb_x_we
                mat_gy[ijc(i_we, j_ns), ijc(ip1, j_ns)] += cfi * jcb_x_ns
                mat_gy[ijc(i_we, j_ns), ijc(im1, j_ns)] -= cfi * jcb_x_ns

        self.grad_x_slow = scp.csr_matrix(mat_gx)
        self.grad_y_slow = scp.csr_matrix(mat_gy)

    def _compute_matrices_csr(self):
        """ compute the matrix form of operators """
        size = self.size_we * self.size_ns
        self.mat_gx = np.zeros((size, size))
        self.mat_gy = np.zeros((size, size))

        # Option 1
        ijc_array = np.arange(self.size_we * self.size_ns)
        iters_i = np.repeat(np.arange(self.size_we), self.size_ns)
        iters_j = np.tile(np.arange(self.size_ns), self.size_we)
        ip1 = np.arange(1, self.size_we + 1)
        jp1 = np.arange(1, self.size_ns + 1)
        im1 = np.arange(-1, self.size_we - 1)
        jm1 = np.arange(-1, self.size_ns - 1)

        cfi_sb = np.ones(self.size_we) * 0.5
        cfj_sb = np.ones(self.size_ns) * 0.5
        if self.periodic_ns:
            im1[0], ip1[-1] = self.size_we - 1, 0
        else:
            im1[0], ip1[-1] = 0, self.size_we - 1
            cfi_sb[0], cfi_sb[-1] = 1., 1.

        if self.periodic_we:
            jm1[0], jp1[-1] = self.size_ns - 1, 0
        else:
            jm1[0], jp1[-1] = 0, self.size_ns - 1
            cfj_sb[0], cfj_sb[-1] = 1., 1.

        cfi = np.repeat(cfi_sb, self.size_ns)
        cfj = np.tile(cfj_sb, self.size_we)

        iters_ip1 = np.repeat(ip1, self.size_ns)
        iters_im1 = np.repeat(im1, self.size_ns)
        iters_jp1 = np.tile(jp1, self.size_we)
        iters_jm1 = np.tile(jm1, self.size_we)
        point_jp1 = ijc_array[iters_i * self.size_ns + iters_jp1]
        point_jm1 = ijc_array[iters_i * self.size_ns + iters_jm1]
        point_ip1 = ijc_array[iters_j + iters_ip1 * self.size_ns]
        point_im1 = ijc_array[iters_j + iters_im1 * self.size_ns]

        column_x = np.concatenate(
            (
                point_jp1[:, None],
                point_jm1[:, None],
                point_ip1[:, None],
                point_im1[:, None],
            ),
            axis=1,
        ).ravel()
        column_y = np.concatenate(
            (
                point_ip1[:, None],
                point_im1[:, None],
                point_jp1[:, None],
                point_jm1[:, None],
            ),
            axis=1,
        ).ravel()
        array_x = np.concatenate(
            (
                (cfj * self.jacob_y_we.ravel())[:, None],
                (-cfj * self.jacob_y_we.ravel())[:, None],
                (-cfi * self.jacob_y_ns.ravel())[:, None],
                (cfi * self.jacob_y_ns.ravel())[:, None],
            ),
            axis=1,
        ).ravel()
        array_y = np.concatenate(
            (
                (cfi * self.jacob_x_ns.ravel())[:, None],
                (-cfi * self.jacob_x_ns.ravel())[:, None],
                (-cfj * self.jacob_x_we.ravel())[:, None],
                (cfj * self.jacob_x_we.ravel())[:, None],
            ),
            axis=1,
        ).ravel()

        grad_x = scp.csr_matrix(
            (array_x, (np.repeat(ijc_array, 4), column_x)),
            shape=(size, size)
        )
        grad_y = scp.csr_matrix(
            (array_y, (np.repeat(ijc_array, 4), column_y)),
            shape=(size, size)
        )
        self.lapl = (grad_x.dot(grad_x) + grad_y.dot(grad_y))  # .tolil()
        self.grad_x_csr = grad_x
        self.grad_y_csr = grad_y

    def _compute_normals(self):
        """
        Compute unit normal vector over the boundaries
        Parameters
        ----------
        Arrays of unit normal vectors : nx and ny.
        """

        x_1d = self.x_coor.ravel()
        y_1d = self.y_coor.ravel()

        normal = np.zeros((self.size_we * self.size_ns, 2))

        # Todo: south and South corners overwritten by east and west
        for bnd_name in ["North", "South", "West", "East"]:
            idx = self.bnd_nodes[bnd_name]

            # Compute dx and dy
            dx_bnd = np.diff(x_1d[idx])
            dy_bnd = np.diff(y_1d[idx])

            # Todo: moyenne artihmetique des normales aux
            # faces adjacentes afin d'avoir les normales
            #aux noeuds + trick perio (moyenne arithmetique
            #avec avec noeud periodique)
            # Compute side effect of diff function for dx and dy
            dx_bnd_end = np.diff(np.roll(x_1d[idx], 1))[-1]
            dy_bnd_end = np.diff(np.roll(y_1d[idx], 1))[-1]

            # Fill normal matrix
            normal[idx[0:-1], 0] = dy_bnd
            normal[idx[-1], 0] = dy_bnd_end
            normal[idx[0:-1], 1] = -dx_bnd
            normal[idx[-1], 1] = -dx_bnd_end

            # Normalize each normal vector
            norm_axis1 = np.linalg.norm(normal[idx, :], axis=1)
            normal[idx] /= norm_axis1.repeat(2).reshape(normal[idx, :].shape)

        # Reshape unit normal vector for output
        print("\n\t\t Metric:\t _compute_normals()\treshape")
        normal_2d_x = normal[:, 0].reshape((self.size_we, self.size_ns))
        normal_2d_y = normal[:, 1].reshape((self.size_we, self.size_ns))

        size_2d = self.size_we, self.size_ns
        if WRITE_BNDY:
            print('\n\t--> Writing "Boundaries.h5"')
            out_h5 = NpArray2Xmf("Boundaries.h5")
            out_h5.create_grid(self.x_coor, self.y_coor, np.zeros(size_2d))

            for bnd_name in ["North", "South", "West", "East"]:
                idx = self.bnd_nodes[bnd_name]
                bnd = np.zeros_like(self.x_coor).ravel()
                bnd[idx] = 1
                out_h5.add_field(bnd.reshape(size_2d), "BC_%s" % bnd_name)

            out_h5.add_field(normal_2d_x, "nx")
            out_h5.add_field(normal_2d_y, "ny")
            out_h5.dump()

        print("\n\t\t Metric:\t _compute_normals()\tstore normals")
        self.n_x = scp.csr_matrix(np.diag(normal_2d_x.ravel()))
        self.n_y = scp.csr_matrix(np.diag(normal_2d_y.ravel()))

        # Gradient normal to the boundary

        # nx_grad_x = np.dot(np.diag(normal_2d_x.ravel()), self.grad_x)
        # ny_grad_y = np.dot(np.diag(normal_2d_y.ravel()), self.grad_y)
        print("\n\t\t Metric:\t _compute_normals()\tcompute nx grad x")
        nx_grad_x = self.n_x.dot(self.grad_x_csr)
        ny_grad_y = self.n_y.dot(self.grad_y_csr)

        print("\n\t\t Metric:\t _compute_normals()\tcompute n grad n")
        self.grad_n_bc = nx_grad_x + ny_grad_y

    def _get_bc_iterators(self):
        """
        Gives node number of boundary patches
        """

        # Flip of numpy array necessary to compute the normals
        i_we = self.size_we - 1
        north = np.flip(
            np.array([(j_idx + self.size_ns * i_we) for j_idx in
                      range(self.size_ns)]))

        # Flip of numpy array necessary to compute the normals
        j_ns = 0
        west = np.flip(
            np.array([(j_ns + self.size_ns * i_idx) for i_idx in
                      range(self.size_we)]))

        i_we = 0
        south = np.array(
            [(j_idx + self.size_ns * i_we) for j_idx in range(self.size_ns)])

        j_ns = self.size_ns - 1
        east = np.array(
            [(j_ns + self.size_ns * i_idx) for i_idx in range(self.size_we)])

        self.bnd_nodes = {}
        self.bnd_nodes["North"] = north
        self.bnd_nodes["West"] = west
        self.bnd_nodes["South"] = south
        self.bnd_nodes["East"] = east

    def diff_ns_2(self, array_):
        """ compute differentiation in direction west (j-direction)"""
        out = np.zeros_like(array_)
        size_j = out.shape[1]

        array_2 = 0.5 * (array_[:, range(2, size_j)] -
                         array_[:, range(0, size_j - 2)])
        out[:, range(1, size_j - 1)] = np.copy(array_2)

        if self.periodic_we:
            out[:, 0] = 0.5 * (array_[:, 1] - array_[:, size_j - 1])
            array_2 = 0.5 * (array_[:, 0] - array_[:, size_j - 2])
            out[:, size_j - 1] = np.copy(array_2)
        else:
            out[:, 0] = array_[:, 1] - array_[:, 0]
            array_2 = array_[:, size_j - 1] - array_[:, size_j - 2]
            out[:, size_j - 1] = np.copy(array_2)

        return out

    def diff_we_2(self, array_):
        """ compute differentiation in direction eta (i-direction)"""
        out = np.zeros_like(array_)
        size_i = out.shape[0]

        array_2 = 0.5 * (array_[range(2, size_i), :] -
                         array_[range(0, size_i - 2), :])
        out[range(1, size_i - 1), :] = np.copy(array_2)

        if self.periodic_ns:
            out[0, :] = 0.5 * (array_[1, :] - array_[size_i - 1, :])

            out[size_i - 1, :] = 0.5 * (array_[0, :] - array_[size_i - 2, :])
        else:
            out[0, :] = array_[1, :] - array_[0, :]
            out[size_i - 1, :] = array_[size_i - 1, :] - array_[size_i - 2, :]

        return out


def set_row_csr(csr, row_idx, new_row):
    """
    Replace a row in a CSR sparse matrix A.

    Parameters
    ----------
    csr: csr_matrix
        Matrix to change
    row_idx: int
        index of the row to be changed
    new_row: np.array
        list of new values for the row of A

    Returns
    -------
    None (the matrix A is changed in place)

    Prerequisites
    -------------
    The row index shall be smaller than the number of rows in A
    The number of elements in new row must be equal to the number of columns in
    matrix A
    """
    assert scp.isspmatrix_csr(csr), 'A shall be a csr_matrix'
    assert row_idx < csr.shape[0], \
        'The row index ({0}) shall be smaller than the number of rows in A ({1})' \
            .format(row_idx, csr.shape[0])
    try:
        n_elements_new_row = len(new_row)
    except TypeError:
        msg = 'Argument new_row shall be a list or numpy array, is now a {0}' \
            .format(type(new_row))
        raise AssertionError(msg)
    n_cols = csr.shape[1]
    assert n_cols == n_elements_new_row, \
        'The number of elements in new row ({0}) must be equal to ' \
        'the number of columns in matrix A ({1})' \
            .format(n_elements_new_row, n_cols)

    idx_start = csr.indptr[row_idx]
    idx_end = csr.indptr[row_idx + 1]
    additional_nnz = n_cols - (idx_end - idx_start)

    # Substitute dense data
    csr.data = np.r_[csr.data[:idx_start], new_row, csr.data[idx_end:]]

    # Correct indices
    csr.indices = np.r_[csr.indices[:idx_start],
                        np.arange(n_cols),
                        csr.indices[idx_end:]]

    # Correct indptr
    csr.indptr = np.r_[csr.indptr[:row_idx + 1],
                       csr.indptr[(row_idx + 1):]
                       + additional_nnz]
