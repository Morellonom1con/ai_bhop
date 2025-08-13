"""Defines a cartesian shell object.
"""

import numpy as np

from kokiy.shell_2d import Shell2D

class CloudShell(Shell2D):
    r"""Base class for Shell of Cloud points.

      This is a degeneration of the usual Shell2D.
      Each point in this point cloud is repeated in the u-direction (u=0 and u=1)
      to keep the shell2D. This avoid many ifs in the code. 
      The U and V connectivity here is just random
      
      The CLoudShell, and its ThickCloudShell expansion 
      should be presented as Points. 
    """
    geom_type = 'cloud'

    def __init__(self, xyz, normals, surf):

        size = xyz.shape[0]
        super().__init__(2,size)
        self.x = np.array([xyz[:,0],xyz[:,0]])
        self.y = np.array([xyz[:,1],xyz[:,1]])
        self.z = np.array([xyz[:,2],xyz[:,2]])
        self.n_x = np.array([normals[:,0],normals[:,0]])
        self.n_y = np.array([normals[:,1],normals[:,1]])
        self.n_z = np.array([normals[:,2],normals[:,2]])
        
        # Keept surface element coherent
        sqrsurf = np.sqrt(surf)
        self.du = np.array([sqrsurf,sqrsurf])
        self.dv = np.array([sqrsurf,sqrsurf])
        self.dwu= self.du
        self.dwv= self.dv
    
    # @property
    # def abs_curv(self):
    #     """override abs curvi"""
    #     return 1.

    def replicate(self, shape):
        """override replicate"""
        raise NotImplementedError("replicate not avialable for CloudShells")