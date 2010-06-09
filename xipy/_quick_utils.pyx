""" -*- python -*- file
"""
__all__ = ['_closest_voxel_i', '_closest_voxel_d']

# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def _closest_voxel_d(
    np.ndarray[np.npy_double, ndim=2] voxels, # nvox x ndim array
    np.ndarray[np.npy_double, ndim=1] location # ndim voxel to match
    ):

    cdef Py_ssize_t nvox = voxels.shape[0]
    cdef Py_ssize_t ndim = voxels.shape[1]
    cdef Py_ssize_t min_k, k = 0, l = 0
    cdef np.npy_double min_dist=1e20, dist, df
    while k < nvox:
        l = 0
        dist = 0.0
        while l < ndim:
            df = voxels[k,l] - location[l]
            dist += (df*df)
            l += 1
        # can short-circuit if distance is 0
        if dist==0.0:
            return k, 0.0
        if dist < min_dist:
            min_dist = dist
            min_k = k
        k += 1

    min_dist = min_dist ** 0.5
    
    return min_k, min_dist

@cython.boundscheck(False)
def _closest_voxel_i(
    np.ndarray[np.npy_int32, ndim=2] voxels, # nvox x ndim array
    np.ndarray[np.npy_int32, ndim=1] location # ndim voxel to match
    ):

    cdef Py_ssize_t nvox = voxels.shape[0]
    cdef Py_ssize_t ndim = voxels.shape[1]
    cdef Py_ssize_t min_k, k = 0, l = 0
    cdef np.npy_int32 min_dist_sq=2**30, dist, df
    while k < nvox:
        l = 0
        dist = 0
        while l < ndim:
            df = voxels[k,l] - location[l]
            dist += (df*df)
            l += 1
        # can short-circuit if distance is 0
        if dist==0:
            return k, 0
        if dist < min_dist_sq:
            min_dist_sq = dist
            min_k = k
        k += 1

    cdef np.npy_double min_dist = min_dist_sq ** 0.5
    
    return min_k, min_dist
