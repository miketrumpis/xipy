import numpy as np
from nipy.core import api as ni_api
## from nipy.algorithms.resample import resample
from xipy.external import resample
from nipy.core.reference.coordinate_map import reorder_output, reorder_input, \
     compose #, drop_io_dim
from scipy import ndimage
from xipy.slicing import SAG, COR, AXI

def fix_analyze_image(img, fliplr=False):
    cmap = img.coordmap
    if fliplr:
        x_idx = cmap.output_coords.index('x')
        cmap.affine[x_idx] *= -1
    if (len(img.shape) < 4) or img.shape[3] != 1:
        return img
    try:
        from nipy.core.reference.coordinate_map import drop_io_dim
    except:
        print 'not fixing this image... NIPY API changed'
        return img
    # affine is a 5x5: need to get rid of row 3 and col 3
    arr = np.asarray(img).reshape(img.shape[:3])
    cmap = drop_io_dim(cmap, 't')
    return ni_api.Image(arr, cmap)

def voxel_size(T):
    T = T[:3,:3]
    return (T**2).sum(axis=0)**.5

def is_diagonal(T):
    T = T[:3,:3]
    vsize = voxel_size(T)
    return vsize.sum() == np.abs(T.diagonal()).sum()

def limits_to_extents(ax_limits):
    """Utility to convert a list of [(xmin, xmax), ... ] pairs to rectangular
    extents (in the Matplotlib AxesImage sense) in SAG, COR, AXI order

    Parameters
    ----------
    ax_limits : iterable of pairs
        3 pairs of (min, max) limits for the SAG, COR, AXI axes

    Returns
    -------
    fig_extents : iterable of box extents
        [ [ umin, umax, vmin, vmax], ... ] for figure coordinates
    """
    sag_extents = ax_limits[COR] + ax_limits[AXI]
    cor_extents = ax_limits[SAG] + ax_limits[AXI]
    axi_extents = ax_limits[SAG] + ax_limits[COR]
    return [sag_extents, cor_extents, axi_extents]

def world_limits(*args):
    """Find the limits of an image volume's box in world space, given the voxel
    to world mapping of the CoordinateMap. The shape of the array is assumed to
    be given in the same order as the input coordinates of the CoordinateMap,
    but the returned limits are in xyz order.

    Parameters
    ----------
    img : a NIPY Image ( if len(args) == 1 )
    coordmap, shape : a NIPY Affine, and 3D grid shape ( if len(args) == 2 )

    Returns
    -------
    a list of 3 2-tuples of the minima/maxima on the x, y, and z axes
    """
    if len(args) == 1:
        coordmap = args[0].coordmap
        shape = args[0].shape
    else:
        coordmap, shape = args
    T = reorder_output(coordmap, 'xyz').affine
    adim, bdim, cdim = shape
    box = np.zeros((8,3), 'd')
    # form a collection of vectors for each 8 corners of the box
    box = np.array([ [0, 0, 0, 1],
                     [adim, 0, 0, 1],
                     [0, bdim, 0, 1],
                     [0, 0, cdim, 1],
                     [adim, bdim, 0, 1],
                     [0, 0, cdim, 1],
                     [adim, 0, cdim, 1],
                     [0, bdim, cdim, 1],
                     [adim, bdim, cdim, 1] ]).transpose()
    box = np.dot(T, box)[:3]
    box_limits = zip(box.min(axis=-1), box.max(axis=-1))
    return box_limits

def vox_limits(img):
    """Find the limits of an image volume's box in its own voxel space--likely
    defined by the axes of the original scan.

    BUT RETURN IN IJK ORDER OR INPUT COORDINATES ORDER???
    """
    T = img.affine
    Ts = T[:3,:3]
    # I think this should preserve directionality ultimately.. worry later
    dv = (Ts**2).sum(axis=0)**.5
    r0 = T[:3,3]
    # since 0 = r0 + Ts*v0, find v0 = solve(Ts, -r0)
    v0 = np.round(np.linalg.solve(Ts, -r0))
    return [(-v0[i]*dv[i], (img.shape[i]-v0[i])*dv[i]) for i in [0,1,2]]

def maximum_world_distance(limits):
    """Find the largest distance between corners in the world volume box.

    Paramters
    ---------
    limits : the {x,y,z} limits of the world volume box

    Returns
    -------
    the distance
    """
    coords = []
    y_limits = limits[1]
    z_limits = limits[2]
    for x in limits[0]:
        for i in [0,1]:
            for j in [0,1]:
                coords.append( [x, y_limits[i], z_limits[j]] )
    coords = np.array(coords)
    diffs = coords[np.newaxis,:,:] - coords[:,np.newaxis,:]
    dist = ( (diffs)**2 ).sum(axis=-1)**.5
    return dist.max()
    
def resample_to_world_grid(img, bbox=None, grid_spacing=None, order=3):
    cmap_xyz = reorder_output(img.coordmap, 'xyz')
    T = cmap_xyz.affine
    if grid_spacing is None:
        # find the (i,j,k) voxel sizes, which should be the norm of the columns
        grid_spacing = list((T[:3,:3]**2).sum(axis=0)**0.5)
    if bbox is None:
        # the extent of the rotated image may cover a larger box
        bbox = world_limits(img)

    box_limits = np.diff(bbox).reshape(3)
    new_dims = np.ceil(box_limits/grid_spacing).astype('i')

    diag_affine = np.diag(list(grid_spacing) + [1])
    diag_affine[:3,3] = np.asarray(bbox)[:,0]
    
    resamp_affine = ni_api.Affine.from_params('ijk', 'xyz', diag_affine)

    # Doing the mapping this way, we don't have to assume what
    # the input space of the Image is like (although we still label it "ijk"
    # in the final resamp_affine)
    mapping = compose(cmap_xyz, img.coordmap.inverse)
    #mapping = compose(resamp_affine, mapping1.inverse)

    new_img = resample.resample(img, resamp_affine, mapping.affine,
                                tuple(new_dims), order=order)

    return new_img


def find_image_threshold(arr, percentile=90., debug=False):
    nbins = 200
    bsizes, bpts = np.histogram(arr.flatten(), bins=nbins)
    # heuristically, this should show up near the middle of the
    # second peak of the intensity histogram
    start_pt = np.abs(bpts - arr.max()/2.).argmin()
    db = np.diff(bsizes[:start_pt])
##     zcross = np.argwhere((db[:-1] < 0) & (db[1:] >= 0)).flatten()[0]
    bval = bsizes[1:start_pt-1][ (db[:-1] < 0) & (db[1:] >= 0) ].min()
    zcross = np.argwhere(bval==bsizes).flatten()[0]
    thresh = (bpts[zcross] + bpts[zcross+1])/2.
    # interpolate the percentile value from the bin edges
    bin_lo = int(percentile * nbins / 100.0)
    bin_hi = int(round(percentile * nbins / 100.0 + 0.5))
    p_hi = percentile - bin_lo # proportion of hi bin
    p_lo = bin_hi - percentile # proportion of lo bin
##     print bin_hi, bin_lo, p_hi, p_lo
    pval = bpts[bin_lo] * p_lo + bpts[bin_hi] * p_hi
    if debug:
        import matplotlib as mpl
        import matplotlib.pyplot as pp
        f = pp.figure()
        ax = f.add_subplot(111)
        ax.hist(arr.flatten(), bins=nbins)
        l = mpl.lines.Line2D([thresh, thresh], [0, .25*bsizes.max()],
                             linewidth=2, color='r')
        ax.add_line(l)
        ax.xaxis.get_major_formatter().set_scientific(True)
        f = pp.figure()
        norm = pp.normalize(0, pval)
        ax = f.add_subplot(211)
        plot_arr = arr
        while len(plot_arr.shape) > 2:
            plot_arr = plot_arr[plot_arr.shape[0]/2]
        ax.imshow(plot_arr, norm=norm)
        ax = f.add_subplot(212)
        simple_mask = (plot_arr < thresh)
        ax.imshow(np.ma.masked_array(plot_arr, mask=simple_mask), norm=norm)
        pp.show()
    
    return thresh, pval

def auto_brain_mask(image_arr, negative=False):
    """ Build a mask function that attempts to segment the brain image
    from the background image.

    Paramters
    ---------
    image_arr : ndarray
        the array of the volume or plane image
    negative : bool, optional
        if negative==True, then return a MaskedArray compatible mask that
        unmasked the brain
    Returns
    -------
    cc_mask : a binary masking function
    """
    thresh, _ = find_image_threshold(image_arr)
    # define a function where the map f(x,y) = True describes the
    # largest connected area where the mean image exceeds the threshold
    fmask = ndimage.binary_fill_holes( image_arr >= thresh )
    labels, n = ndimage.label(fmask)
    lsizes = [ (labels==i).sum() for i in xrange(1, n+1) ]
    max_label = np.argmax(lsizes)+1
    cc_mask = (labels==max_label)
    return np.logical_not(cc_mask) if negative else cc_mask

def signal_array_to_masked_vol(sig, vox_indices,
                               grid_shape=[],
                               prior_mask=None,
                               **ma_kw):
    """Make a 3D array representing a mask for valid voxels locations,
    given an array of voxel indices.

    Parameters
    ----------
    sig : ndarray
        nvox x [num_measures] array of signal measurements, whose spatial
        order is given by the corresponding (following) voxel array
    vox_indices : ndarray
        nvox x 3 array of voxel indices (NOT voxel locations in MNI space)
    grid_shape : list (optional)
        an list of dimension extents, eg [imax, jmax, kmax]
    prior_mask : array-like (optional)
        an nvox length array indicating points to mask in the final volume
        (True = unmasked, opposite of MaskedArray convention)
    ma_kw : dict
        Keyword arguments for np.ma.masked_array

    Returns
    -------
    s_masked : a numpy MaskedArray, with non-map voxels masked out
    """
    if not grid_shape:
        if not len(sig):
            return np.ma.masked_array(np.empty((1,1,1)),
                                      mask=np.ones((1,1,1), dtype=np.bool),
                                      **ma_kw)
        ix, jx, kx = map(lambda x: x+1, vox_indices.max(axis=0))
    else:
        ix, jx, kx = grid_shape

    vmask = np.ones((ix,jx,kx) + sig.shape[1:], np.bool)
    s = np.zeros((ix,jx,kx) + sig.shape[1:], sig.dtype)

    if prior_mask is not None:
        i, j, k = vox_indices[prior_mask].T
        sig = sig[prior_mask]
    else:
        i, j, k = vox_indices.T

    flat_idx = i*(jx*kx) + j*kx + k

    if sig.shape[1:]:
        # then we need to add more indices
        vx = np.product(sig.shape[1:])
        v = np.arange(vx)
        flat_idx *= vx
        flat_idx = (flat_idx[:,None] + v).reshape(-1)

    s.flat[flat_idx] = sig
    vmask.flat[flat_idx] = False
    return np.ma.masked_array(s, mask=vmask, **ma_kw)


## class FlatSubVolume(object):
##     def __init__(self, map_scalars, map_voxels):
##         self.map_scalars = map_scalars
##         self.map_voxels = map_voxels

##     def map_to_volume(self, coordmap, dims=(), array_order='C'):
##         """Map self to a MaskedArray volume.

##         Parameters
##         ----------
##         coordmap : NIPY CoordinateMap
##             a coordinate mapping from array coordinates to world coordinates,
##             where the world coordinate space is that from which map_voxels
##             are drawn.

##         Returns
##         -------
##         a masked array
##         """
##         map_indices = coordmap.inverse(self.map_voxels)
##         return 
