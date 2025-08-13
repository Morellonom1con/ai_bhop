"""Extract features from a curve"""

import numpy as np

def reptile(arr, axis, pos):
    """expand  "row" POS from axis AXIS of an array ARR everywhere else

    Args:
        arr (np.array): source array
        axis (int): index of axis to expand
        pos (int): index of element to expand
        
    Returns:
        np.array: Expanded array

    >>> a = np.array([[0, 1, 0, 0, 0],
                      [0, 1, 1, 1, 1],
                      [0, 1, 2, 3, 4]])
    >>> reptile(a, axis=1, pos=3)
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [3, 3, 3, 3, 3]])


    """
    sss = [slice(None) for _ in range(arr.ndim)]
    sss[axis] = np.s_[pos:pos+1:]
    return np.repeat(arr[tuple(sss)], arr.shape[axis], axis=axis)


def amax_keepshape(arr, axis):
    """expand  "row" POS from axis AXIS of an array ARR everywhere else

    Args:
        arr (np.array): source array
        axis (int): index of axis to expand
        
    Returns:
        np.array: amax on the same shape

    >>> a = np.array([[0, 1, 0, 0, 0],
                      [0, 1, 1, 1, 1],
                      [0, 1, 2, 3, 3]])
    >>> amax_keepshape(a, axis=1, pos=3)
    array([[1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1],
          [3, 3, 3, 3, 3]])


    """

    return np.repeat(arr.max(axis=axis, keepdims=True), arr.shape[axis], axis=axis)


def amin_keepshape(arr, axis):
    """expand  "row" POS from axis AXIS of an array ARR everywhere else

    Args:
        arr (np.array): source array
        axis (int): index of axis to expand
        
    Returns:
        np.array: amin on the same shape

    >>> a = np.array([[0, 1, 0, 0, 0],
                     [0, 1, -1, 1, 1],
                     [1, 1, 2, 3, 4]])
    >>> amin_keepshape(a, axis=1, pos=3)
    array([[0, 0, 0, 0, 0],
           [-1, -1, -1, -1, -1],
           [1, 1, 1, 1, 1]])

    """
    return np.repeat(arr.min(axis=axis, keepdims=True), arr.shape[axis], axis=axis)





def _end_match(arr_, kappa=0.2, axis=0, end_index=None):
    """Return a linear approx of the  input curve,
    matching exactly at x=x_ref and y=(1-kappa) x_ref. 

    Args:
        arr_ (np.array): source array
        kappa (float): ]0,1[, position of the first point to place the linear match
        axis (int): index of axis to search
        end_index (int):Index of second point to search the linear match (If None, last index)
        
        
    Returns:
        np.array: linear match of source array

        value                                                   
        ^                                                   
        |                                                   
        |               |\                                  
        |   *          /  \                                 
        |         *   /    \                                
        |             |*    -                               
        |            /      *---                            
        |           /           \*---                       
        |          /             .   \----                  
        |          |             .     -  \----
        ----------/              .             \----
        |                        .                  \-*
        |                        .                    .    *
        |                        .                    .
        |                        .                    .
        |                        .                    .
        |----------------------------------------------> axis
                    int((1-kappa)end_index)            end_index
    """
    size =  arr_.shape[axis]

    if end_index is not None:
        idx2 = end_index
    else:
        idx2 = size-1
    idx1 = min(int((1-kappa)*idx2), idx2-1)
    
    y2 = reptile(arr_,axis, idx2)
    y1 = reptile(arr_,axis, idx1)
    
    slope = (y2-y1)/(idx2-idx1)
    adjust = y2 - slope*idx2
    
    x_vals = np.indices(arr_.shape)[axis]
    return slope*x_vals + adjust


def _mask_deviation(arr_, eps, axis, gain=1.):
    """Create a mask true where a fluctuation array far from zero.

    Args:
        arr_ (np.array): source array
        eps (float): ]0,1[ relative fluctuation thresold (0.1 is 10% of max curve amplitude)
        axis (int): index of axis to search
        gain (float): >0 increase or decrease the with of the layer found 
        
        
        
    Returns:
        np.array: masked array equal to one on the fluctuation side.

        value                                                                                
        ^                                                                                
        |                                                                               
        |  x                                                                            
        | x  x                                                                          
   eps  |...............................................                                 
        |x    x                                                                         
        x       x                                                                      
    0    -----------x. x. x. x. x. x. x. x. x  x. x -----> axis                              
                                                                                       
                                                                                       
    -eps ................................................                                 
                                                                                       
                                                                         
        111111000000000000000000000000000000000000000     

    Notes:
        The fluctuation array must vanish to zero toward the end of AXIS
        The thresold EPS is relative to the local ampliture of the 1D vector.
        The wall node is always one in the outpud

                                                                
    """
    
    x_vals = np.indices(arr_.shape)[axis]
    max_index = arr_.shape[axis]-1
    wall = np.where(x_vals==0, 1, 0)
    
    arr_calib = arr_ + 1.e-12*wall
    amp = amax_keepshape(arr_calib, axis=axis)-amin_keepshape(arr_calib, axis=axis)
    
    mask = np.where(np.abs(arr_calib)>eps*amp, 1, 0)
    wh_ = np.where(mask==1, x_vals, 0)
    thresold = amax_keepshape(wh_, axis=axis)
    thresold = np.rint(thresold*gain).astype(int) # apply the gain
    deviation_mask = np.where(x_vals<=np.clip(thresold,0,max_index), 1, 0)
    return deviation_mask

def mask_boundary_layer(source, eps=0.1, gain=1., axis=0):
    """Identify the cells relative to a boundary layer in a numpy multi-d array.
    If no fluctuation detected, only the "wall" node is set to 1.

    Args:
        source (np.array): source array
        eps (float): ]0,1[ relative fluctuation thresold (0.1 is 10% of max curve amplitude)
        gain (float): >0 increase of decrease the width of the layer found 
        axis (int): index of axis to search
        
        
    Returns:
        np.array: masked array equal to one on the fluctuation side.
    
    >>> a = np.array(
                   [[ 0,  1,  0,  0,  0]
                    [ 0,  1,  1,  1,  1]
                    [ 0,  3,  2,  3,  4]
                    [ 0,  1, -2,  3,  4]
                    [ 0,  0,  0,  0,  0]
                    [ 0,  1,  2,  3,  4]]
        )
    >>> mask_boundary_layer(ource, eps=0.1, axis=1)
    array(
        [[1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]]
    )
    
    """

    endm =  _end_match(source, axis=axis)
    fluctu = source - endm
    mask = _mask_deviation(fluctu, eps=eps, axis=axis, gain=gain)
    return mask