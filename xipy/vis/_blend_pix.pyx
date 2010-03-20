""" -*- python -*- file
"""

import numpy as np
cimport numpy as cnp

def resample_and_blend(base_arr, base_dr, base_r0,
                       over_arr, over_dr, over_r0):
    """ Taking two specs for VTKImageData-like data (array, spacing, offset),
    resample the overlying array into the base array with RGBA pixel blending
    (alpha blending).

    Paramters
    ---------
    base_arr : ndarray, dtype=uint8, len(shape) > 2
        The array of RGBA byte values (IE, scalar values in [0,255]). The
        color component dimension is last.
    base_dr : iterable
        The grid spacing of the base array, for each grid dimension
    base_r0 : iterable
        The offset coordinates of the base array, for each grid dimension
    over_arr : ndarray, dtype=uint8, len(shape) > 2
        RGBA byte values for the overlying array
    over_dr : iterable
        The grid spacing of the overlying array, for each grid dimension
    over_r0 : iterable
        The offset coordinates of the overlying array, for each grid dimension

    Returns
    -------
    Blends the two arrays into the base_arr
    """
    # Make sure the arrays are safe for the C++ code
    assert base_arr.shape[-1] == 4 and over_arr.shape[-1] == 4, \
           'Arrays must have RGBA components in the last dimension'
    assert base_arr.dtype.char == 'B' and over_arr.dtype.char == 'B', \
           'Array types must be byte-valued'
    assert len(base_arr.shape) == (len(base_dr)+1) and \
           len(base_arr.shape) == (len(base_r0)+1), \
           'The orientation specs (dr and r0) for the base array do not '\
           'seem to match the array'
    assert len(over_arr.shape) == (len(over_dr)+1) and \
           len(over_arr.shape) == (len(over_r0)+1), \
           'The orientation specs (dr and r0) for the base array do not '\
           'seem to match the array'
    # Make sure the parameters conform to the expected shapes and sizes
    bshape1 = base_arr.shape
    bshape = list(base_arr.shape)
    base_dr = list(base_dr); base_r0 = list(base_r0)
    while len(bshape) < 4:
        bshape.insert(0,1)
        base_dr.insert(0,1)
        base_r0.insert(0,0)
    base_arr.shape = tuple(bshape)
    b_dr = np.array(base_dr)
    b_r0 = np.array(base_r0)

    oshape1 = over_arr.shape
    oshape = list(over_arr.shape)
    over_dr = list(over_dr); over_r0 = list(over_r0)
    while len(oshape) < 4:
        oshape.insert(0,1)
        over_dr.insert(0,1)
        over_r0.insert(0,0)
    over_arr.shape = tuple(oshape)
    o_dr = np.array(over_dr)
    o_r0 = np.array(over_r0)

    cdef int i, j, k, ii, jj, kk
    cdef cnp.npy_double wcoord
    cdef cnp.npy_uint bpr, bpg, bpb, bpa, opr, opg, opb, opa
    for i in range(bshape[0]):
        for j in range(bshape[1]):
            for k in range(bshape[2]):
                wcoord = i*b_dr[0] + b_r0[0]
                ii = int( (wcoord - o_r0[0]) / o_dr[0] )
                wcoord = j*b_dr[1] + b_r0[1]
                jj = int( (wcoord - o_r0[1]) / o_dr[1] )
                wcoord = k*b_dr[2] + b_r0[2]
                kk = int( (wcoord - o_r0[2]) / o_dr[2] )
                if ii>=0 and ii<oshape[0] and \
                   jj>=0 and jj<oshape[1] and \
                   kk>=0 and kk<oshape[2]:
                    bpr, bpg, bpb, bpa = base_arr[i,j,k,:]
                    opr, opg, opb, opa = over_arr[ii,jj,kk,:]
                    base_arr[i,j,k,0] = (((opr-bpr)*opa + (bpr<<8))>>8)
                    base_arr[i,j,k,1] = (((opg-bpg)*opa + (bpg<<8))>>8)
                    base_arr[i,j,k,2] = (((opb-bpb)*opa + (bpb<<8))>>8)
                    # what to do about alpha?
    #done
    base_arr.shape = bshape1
    over_arr.shape = oshape1
                    
                   
                
