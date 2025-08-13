###Testing a patch mapper 2D feature"""

import numpy as np
from arnica.utils import NpArray2Xmf

__all__ = ["AxiPointCloud"]

class AxiPointCloud():
    """Handle Axysymmetric points clouds as 1D meshs 
    with a dict of variables

    
    """

    def __init__(self,xcoor, ycoor, zcoor, name="Unnamed", vars=None, theta_range=None):
        """ startup class
        
        Parameters:
        ===========
        xcoor, ycoor, zcoor: coordinates 
            1D np array in meters
        name: name of the object
            string
        vars: variables associated to pointcloud
            dictionnary of 1D np.arrays
        theta_range: override the theta range, in radians
            float
        """

        self.name = name

        self.x = np.array(xcoor)
        self.y = np.array(ycoor)
        self.z = np.array(zcoor)
        if theta_range is None:
            self.recompute_theta_range_from_coords()
        else:
            self.theta_range=theta_range

        self.vars = dict()
        if vars is not None:
            for var in vars:
                self.vars[var] = np.array(vars[var])


    def __str__(self):
        """String representation of pt cloud"""

        def _print_data(name, arr):
            """utility to print arrays"""
            str_ = "-"+name+"----"
            str_ += "\nshape :"+ str(arr.shape)
            str_ += "\nmin/max :"+ str(arr.min())+"/"+str(arr.max())
            return str_

        str_ = f'\nAxiPointCloud object "{self.name}"'
        str_ += "\n\n Angle section (deg.):" + str(np.rad2deg(self.theta_range))
        str_ += "\n\n Coordinates:"
        str_ += "\n"+ _print_data("X coor (m)",self.x)
        str_ += "\n"+ _print_data("Y coor (m)",self.y)
        str_ += "\n"+ _print_data("Z coor (m)",self.z)
        str_ += "\n"+ _print_data("R coor (m)",self.rad())
        str_ += "\n"+ _print_data("Theta coor (deg)",np.rad2deg(self.theta()))
        str_ += "\n\n Data:"
        for key in self.vars:
            str_ += "\n"+ _print_data(key,self.vars[key])
        return str_

    def xyz(self):
        """Stacked version numpy of coordinates
        
        shape: (n, 3)"""
        return np.stack((self.x, self.y, self.z), axis=1)

    def vars_stack(self):
        """Stacked version numpy of variables
        
        shape: (n, k)"""

        keys = list(self.vars.keys())
        stack = self.vars[keys[0]]

        for key in keys[1:]:
            stack = np.stack((stack,self.vars[key] ), axis=1)
        return stack



    def rad(self):
        """Return radius np array from Y and Z"""
        return np.hypot(self.y,self.z)

    def theta(self):
        """Return theta np array (radians)
        
        - range -pi/pi
            - 0 on the y+ axis (z=0, y>0)
            spanning -pi to pi
                        0pi=0deg
                        Y
                        ^
                        |
                        |
        -0.5pi=-90deg     o------>Z   0.5pi=90deg
        """

        return np.arctan2(self.z,self.y)

    def recompute_theta_range_from_coords(self):
        """recompute theta range"""
        theta = self.theta()
        self.theta_range = theta.max() - theta.min()

    def rotate(self, shift_angle):
        """ rotate around x
        
        :param shift_angle: angle in radians
            float
        """

        radius = self.rad()
        theta = self.theta() + shift_angle

        self.y = radius * np.cos(theta)
        self.z = radius * np.sin(theta)


    def dupli_rotate(self, repeat=1):
        """ duplicate by rotation around x, in radians
        
        :param repeat: number or repetitions
            integer
        """

        radius = self.rad()
        theta = self.theta()
        for _ in range(repeat):
            theta += self.theta_range
            self.y = np.concatenate((self.y, radius * np.cos(theta)))
            self.z = np.concatenate((self.z, radius * np.sin(theta)))

        self.x = np.tile(self.x, repeat+1)
        for arr in self.vars:
            self.vars[arr] = np.tile(self.vars[arr], repeat+1)

        self.theta_range *= repeat+1

    def dump(self, filename):
        """Dump an XDMF file of the pointcloud"""

        stream = NpArray2Xmf(filename)
        stream.create_grid(self.x, self.y, self.z)
        for key in self.vars:
            stream.add_field(self.vars[key], key)
        stream.dump()
