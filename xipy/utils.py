import scipy.io as sio
import numpy as np
import os

from xipy._quick_utils import _closest_voxel_i, _closest_voxel_d

def with_attribute(a):
    def dec(f):
        def runner(obj, *args, **kwargs):
            if not getattr(obj, a, False):
                return
            return f(obj, *args, **kwargs)
        # copy f's info to runner
        for attr in ['func_doc', 'func_name']:
            setattr(runner, attr, getattr(f, attr))
        return runner
    return dec

def voxel_index_list(shape, order='ijk'):
    """From an array shape, return a list of voxel index-coordinates

    Parameters
    ----------
    shape : tuple
        the array shape
    order : str, optional
        Indicates whether the coordinate ordering should go 'ijk', or 'kji'.
        In either case, the i coordinate varies fasted,
        followed by [j, [k, [ ... ]]]

    Returns
    -------
    an ( nvox x len(shape) ) array of voxel index coordinates

    Examples
    --------
    >>> utils.voxel_index_list((2,3), order='ijk')
    array([[0, 0],
           [1, 0],
           [0, 1],
           [1, 1],
           [0, 2],
           [1, 2]])
    >>> utils.voxel_index_list((2,3), order='kji')
    array([[0, 0],
           [0, 1],
           [0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])
    """
    if order=='ijk':
        arr = np.indices(shape[::-1])[::-1]
    else:
        arr = np.indices(shape)
    return np.array( [a.flatten() for a in arr] ).transpose()

def coord_list_to_mgrid(coords, shape, order='ijk'):
    """From a voxel coordinates list, make a meshgrid array

    Parameters
    ----------
    coords : np.product(shape) x ndim ndarray
        the voxel coordinate list -- MUST have coordinates for every
        point in the volume specified by shape
    shape : tuple
        the volume's shape
    order : str, optional
        Indicates whether the coordinate ordering is 'ijk', or 'kji'.
        In either case, the i coordinate varies fasted,
        followed by [j, [k, [ ... ]]]

    Returns
    -------
    a meshgrid representation of the coordinates, shaped (ndim, ni, nj, nk, ...)
    or (ndim, [...], nk, nj, ni)

    Examples
    --------
    >>> coords_list = utils.voxel_index_list((2,3), order='ijk')
    >>> coords_list
    array([[0, 0],
           [1, 0],
           [0, 1],
           [1, 1],
           [0, 2],
           [1, 2]])
    >>> mgrid = utils.coord_list_to_mgrid(coords_list, (2,3), order='ijk')
    >>> mgrid
    array([[[0, 0, 0],
            [1, 1, 1]],

           [[0, 1, 2],
            [0, 1, 2]]])
    """
    ncoords = coords.shape[0]
    if np.product(shape) != ncoords:
        raise ValueError(
"""A fully specified coordinate list must be provided"""
    )
    c_t = coords.transpose()
    if order=='ijk':
        return np.array( [a.reshape(shape[::-1]).transpose() for a in c_t] )
    else:
        return np.array( [a.reshape(shape) for a in c_t] )

def closest_voxel(voxels, location):
    assert voxels.ndim==2, 'Voxel list mis-shapen'
    nd = voxels.shape[1]
    try:
        location = np.asarray(location).reshape(nd)
    except:
        raise ValueError('Location argument has the '\
                         'wrong dimensionality: %s'%repr(location))
    if voxels.dtype not in np.sctypes['float'] + np.sctypes['int'] or \
           location.dtype not in np.sctypes['float'] + np.sctypes['int']:
        raise ValueError("Can't lookup coordinates with this "\
                         "dtype: %s, %s"%(repr(voxels.dtype),
                                          repr(location.dtype)))

    # branch on dtype of voxel list    
    func, dt = (_closest_voxel_d, 'd') \
               if voxels.dtype in np.sctypes['float'] \
               else (_closest_voxel_i, 'i')

    if voxels.dtype.char != dt:
        voxels = voxels.astype(dt)
    if location.dtype.char != dt:
        location = location.astype(dt)
    return func(voxels, location)

class MNI_to_Talairach_db(object):

    def __init__(self, db='icbm'):
        import xipy
        where = os.path.abspath(xipy.__file__)
        where = os.path.dirname(where)
        db_file = os.path.join(where, 'resources/MNIicbm.mat')
        db_arr = sio.loadmat(db_file, struct_as_record=True)['MNIdm'][0,0]
        labels = db_arr['labels']
        self.locations = db_arr['coords'].astype('d')
        self.lut = db_arr['data'].astype('h')
        table = []
        for row in labels:
            table.append([str(e[0,0][0])
                          if e[0,0].shape != (0,) else '' for e in row ])
        self.table = table

    def __call__(self, loc):
        try:
            loc = np.asarray(loc).reshape(3)
        except:
            raise ValueError('Location argument has the '\
                             'wrong dimensionality: %s'%repr(loc))
        loc = loc.astype('d')
        idx, dist = closest_voxel(self.locations, loc)
        if dist > 2:
            return [''] * 5
        table_idx = self.lut[idx]
        return [self.table[i][table_idx] for i in xrange(len(self.table))]

        
