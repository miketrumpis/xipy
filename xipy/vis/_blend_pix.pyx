""" -*- python -*- file
"""

# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython

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
    b_dr = np.array(base_dr, dtype='d')
    b_r0 = np.array(base_r0, dtype='d')

    oshape1 = over_arr.shape
    oshape = list(over_arr.shape)
    over_dr = list(over_dr); over_r0 = list(over_r0)
    while len(oshape) < 4:
        oshape.insert(0,1)
        over_dr.insert(0,1)
        over_r0.insert(0,0)
    over_arr.shape = tuple(oshape)
    o_dr = np.array(over_dr, dtype='d')
    o_r0 = np.array(over_r0, dtype='d')
    over_arr_r = resize_rgba_array(tuple(bshape[:3]), over_arr,
                                   o_dr, o_r0,
                                   b_dr, b_r0)
    over_arr_r.shape = ( np.prod(bshape[:3]), 4 )
    base_arr.shape = ( np.prod(bshape[:3]), 4 )
    blend_same_size_arrays(base_arr, over_arr_r)
##     blend_arrays(base_arr, b_dr, b_r0, over_arr, o_dr, o_r0)
    #done
    base_arr.shape = bshape1
    over_arr.shape = oshape1

## @cython.profile(True)
cdef inline Py_ssize_t in_bounds(Py_ssize_t i, Py_ssize_t N):
    if i<0: return 0
    if i>=N: return N-1
    return i

## @cython.profile(True)
cdef inline int is_in_bounds(Py_ssize_t i, Py_ssize_t N):
    if i<0 or i>=N: return 0
    return 1

## cdef inline void fill_bg_sub(Py_ssize_t N,
##                              np.ndarray[np.npy_ubyte, ndim=1] subarr,
##                              np.ndarray[np.npy_ubyte, ndim=1] bg):
##     cdef Py_ssize_t i = 0
##     while i<N*4:
##         subarr[i] = bg[0]; i+=1
##         subarr[i] = bg[1]; i+=1
##         subarr[i] = bg[2]; i+=1
##         subarr[i] = bg[3]; i+=1

## @cython.profile(True)
@cython.boundscheck(False)
def resize_rgba_array(
    new_shape, # new shape tuple
    np.ndarray[np.npy_ubyte, ndim=4] arr, # flattened array
    np.ndarray[np.npy_double, ndim=1] dr, # grid spacing
    np.ndarray[np.npy_double, ndim=1] r0, # offset vector
    np.ndarray[np.npy_double, ndim=1] bdr, # grid spacing
    np.ndarray[np.npy_double, ndim=1] br0 # offset vector
    ): 

    cdef Py_ssize_t ni = new_shape[0], nj = new_shape[1], nk = new_shape[2]
    cdef Py_ssize_t oi = arr.shape[0], oj = arr.shape[1], ok = arr.shape[2]
    cdef Py_ssize_t i, j, k, ii, jj, kk, u, nu
    cdef np.ndarray[np.npy_ubyte, ndim=4] n_arr = np.zeros(new_shape+(4,),
                                                           dtype=np.uint8)
    cdef np.ndarray bg_color = np.array([255,255,255,0], dtype=np.uint8)
    cdef np.npy_double wcoord
    for i in xrange(ni):
        wcoord = i*bdr[0] + br0[0]
        ii = <Py_ssize_t>( (wcoord - r0[0])/dr[0] )
##         ii = in_bounds(ii, oi)
        if not is_in_bounds(ii,oi):
            continue
##         if not in_bounds(ii, oi):
##             # can't do this inline yet, apparently
## ##             fill_bg_sub(nj*nk, n_arr[i].flatten(), bg_color)
##             u = 0; nu = 4*nj*nk
##             t_arr = n_arr[i].flatten()
##             while u < nu:
##                 t_arr[u] = bg_color[0]; u+=1
##                 t_arr[u] = bg_color[1]; u+=1
##                 t_arr[u] = bg_color[2]; u+=1
##                 t_arr[u] = bg_color[3]; u+=1
##             continue
        for j in xrange(nj):
            wcoord = j*bdr[1] + br0[1]
            jj = <Py_ssize_t>( (wcoord - r0[1])/dr[1] )
##             jj = in_bounds(jj, oj)
            if not is_in_bounds(jj,oj):
                continue
##             if not in_bounds(jj, oj):
## ##                 fill_bg_sub(nk, n_arr[i,j].flatten(), bg_color)
##                 u = 0; nu = 4*nk
##                 t_arr = n_arr[i,j].flatten()
##                 while u < nu:
##                     t_arr[u] = bg_color[0]; u+=1
##                     t_arr[u] = bg_color[1]; u+=1
##                     t_arr[u] = bg_color[2]; u+=1
##                     t_arr[u] = bg_color[3]; u+=1
##                 continue
            for k in xrange(nk):
                wcoord = k*bdr[2] + br0[2]
                kk = <Py_ssize_t>( (wcoord - r0[2])/dr[2] )
##                 kk = in_bounds(kk, ok)
                if not is_in_bounds(kk,ok):
                    continue
##                 if not in_bounds(kk,ok):
## ##                     fill_bg_sub(1, n_arr[i,j,k], bg_color)
##                     t_arr = n_arr[i,j,k]
##                     t_arr[0] = bg_color[0];
##                     t_arr[1] = bg_color[1];
##                     t_arr[2] = bg_color[2];
##                     t_arr[3] = bg_color[3];
##                     continue
##                 if not (in_bounds(ii,oi) and \
##                         in_bounds(jj,oj) and \
##                         in_bounds(kk,ok)):
##                     n_arr[i,j,k,0] = bg_color[0];
##                     n_arr[i,j,k,1] = bg_color[1];
##                     n_arr[i,j,k,2] = bg_color[2];
##                     n_arr[i,j,k,3] = bg_color[3];
##                     continue
                n_arr[i,j,k,0] = arr[ii,jj,kk,0]
                n_arr[i,j,k,1] = arr[ii,jj,kk,1]
                n_arr[i,j,k,2] = arr[ii,jj,kk,2]
                n_arr[i,j,k,3] = arr[ii,jj,kk,3]                
    return n_arr

## @cython.boundscheck(False)
## def resize_lookup_array(
##     new_shape, # new shape tuple
##     int i_bad,
##     np.ndarray[np.npy_int32, ndim=3] arr, # flattened array
##     np.ndarray[np.npy_double, ndim=1] dr, # grid spacing
##     np.ndarray[np.npy_double, ndim=1] r0, # offset vector
##     np.ndarray[np.npy_double, ndim=1] bdr, # grid spacing
##     np.ndarray[np.npy_double, ndim=1] br0 # offset vector
##     ): 

##     cdef Py_ssize_t ni = new_shape[0], nj = new_shape[1], nk = new_shape[2]
##     cdef Py_ssize_t oi = arr.shape[0], oj = arr.shape[1], ok = arr.shape[2]
##     cdef Py_ssize_t i, j, k, ii, jj, kk, u, nu
##     cdef np.ndarray[np.npy_int32, ndim=3] n_arr = np.ones(new_shape,
##                                                           dtype=np.int32)
##     n_arr *= i_bad
##     cdef np.npy_double wcoord
##     for i in xrange(ni):
##         wcoord = i*bdr[0] + br0[0]
##         ii = <Py_ssize_t>( (wcoord - r0[0])/dr[0] )
##         if not is_in_bounds(ii,oi):
##             continue
##         for j in xrange(nj):
##             wcoord = j*bdr[1] + br0[1]
##             jj = <Py_ssize_t>( (wcoord - r0[1])/dr[1] )
##             if not is_in_bounds(jj,oj):
##                 continue
##             for k in xrange(nk):
##                 wcoord = k*bdr[2] + br0[2]
##                 kk = <Py_ssize_t>( (wcoord - r0[2])/dr[2] )
##                 if not is_in_bounds(kk,ok):
##                     continue
##                 n_arr[i,j,k] = arr[ii,jj,kk]
##     return n_arr

@cython.boundscheck(False)
def resize_lookup_array(
    new_shape,
    int i_bad,
    np.ndarray[np.npy_int32, ndim=3] src, # source array to sample from
    np.ndarray[np.npy_double, ndim=1] scale, # matrix diagonal
    np.ndarray[np.npy_double, ndim=1] shift, # coordinate translations
    ):

    cdef Py_ssize_t ni = new_shape[0], nj = new_shape[1], nk = new_shape[2]
    cdef Py_ssize_t oi = src.shape[0], oj = src.shape[1], ok = src.shape[2]
    cdef Py_ssize_t i, j, k, ii, jj, kk
    cdef np.ndarray[np.npy_int32, ndim=3] n_arr = np.ones(new_shape,
                                                          dtype=np.int32)
    n_arr.fill(i_bad)
##     cdef np.npy_double wcoord
    cdef np.npy_double s0 = scale[0], s1 = scale[1], s2 = scale[2]
    cdef np.npy_double t0 = shift[0], t1 = shift[1], t2 = shift[2]
    for i in xrange(ni):
        ii = <Py_ssize_t>(i * s0 + t0)
        if not is_in_bounds(ii,oi):
            continue
        for j in xrange(nj):
            jj = <Py_ssize_t>(j * s1 + t1)
            if not is_in_bounds(jj,oj):
                continue
            for k in xrange(nk):
                kk = <Py_ssize_t>(k * s2 + t2)
                if not is_in_bounds(kk,ok):
                    continue
                n_arr[i,j,k] = src[ii,jj,kk]
    return n_arr
    
    
@cython.boundscheck(False)
def blend_same_size_arrays(np.ndarray[np.npy_ubyte, ndim=2] b_arr,
                           np.ndarray[np.npy_ubyte, ndim=2] o_arr):
    cdef Py_ssize_t i, n_pt = b_arr.shape[0]
    for i in xrange(n_pt):
        b_arr[i,0] = blend(b_arr[i,0], o_arr[i,0], o_arr[i,3])
        b_arr[i,1] = blend(b_arr[i,1], o_arr[i,1], o_arr[i,3])
        b_arr[i,2] = blend(b_arr[i,2], o_arr[i,2], o_arr[i,3])
        b_arr[i,3] = blend_alpha(b_arr[i,3], o_arr[i,3])

@cython.boundscheck(False)
def blend_arrays(np.ndarray[np.npy_ubyte, ndim=4] b_arr,
                 np.ndarray[np.npy_double, ndim=1] b_dr,
                 np.ndarray[np.npy_double, ndim=1] b_r0,
                 np.ndarray[np.npy_ubyte, ndim=4] o_arr,
                 np.ndarray[np.npy_double, ndim=1] o_dr,
                 np.ndarray[np.npy_double, ndim=1] o_r0):
##     cdef int i, j, k, ii, jj, kk
    cdef Py_ssize_t i, j, k, ii, jj, kk    
    cdef np.npy_double wcoord
##     cdef np.npy_uint bpr, bpg, bpb, bpa, opr, opg, opb, opa
##     cdef np.ndarray[np.uint8_t, ndim=1] bpx = np.empty((4,), dtype=np.uint8)
##     cdef np.ndarray[np.uint8_t, ndim=1] opx = np.empty((4,), dtype=np.uint8)
    cdef Py_ssize_t ni=b_arr.shape[0], nj=b_arr.shape[1], nk=b_arr.shape[2]
    cdef Py_ssize_t oni=o_arr.shape[0], onj=o_arr.shape[1], onk=o_arr.shape[2]
    for i in xrange(ni):
        for j in xrange(nj):
            for k in xrange(nk):
                wcoord = i*b_dr[0] + b_r0[0]
                ii = int( (wcoord - o_r0[0]) / o_dr[0] )
                wcoord = j*b_dr[1] + b_r0[1]
                jj = int( (wcoord - o_r0[1]) / o_dr[1] )
                wcoord = k*b_dr[2] + b_r0[2]
                kk = int( (wcoord - o_r0[2]) / o_dr[2] )
                if ii>=0 and ii<oni and \
                   jj>=0 and jj<onj and \
                   kk>=0 and kk<onk:
                    b_arr[i,j,k,0] = blend(b_arr[i,j,k,0], o_arr[ii,jj,kk,0],
                                           o_arr[ii,jj,kk,3])
                    b_arr[i,j,k,1] = blend(b_arr[i,j,k,1], o_arr[ii,jj,kk,1],
                                           o_arr[ii,jj,kk,3])
                    b_arr[i,j,k,2] = blend(b_arr[i,j,k,2], o_arr[ii,jj,kk,2],
                                           o_arr[ii,jj,kk,3])
                    # what to do about alpha?
                    b_arr[i,j,k,3] = blend_alpha(b_arr[i,j,k,3],
                                                 o_arr[ii,jj,kk,3])
                    
cdef inline np.npy_ubyte blend(np.npy_ubyte c1, np.npy_ubyte c2,
                               np.npy_ubyte a2):
    cdef np.npy_uint color1 = c1, color2 = c2, alpha2 = a2
    return <np.npy_ubyte>(((color2-color1)*alpha2 + (color1<<8))>>8)

cdef inline np.npy_ubyte blend_alpha(np.npy_ubyte a1, np.npy_ubyte a2):
    cdef np.npy_uint alpha1 = a1, alpha2 = a2
    return <np.npy_ubyte>((alpha2 + alpha1) - ((alpha2*alpha1 + 255) >> 8))
    
    

