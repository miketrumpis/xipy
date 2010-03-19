import numpy as np
import matplotlib as mpl

def blend_two_images(base_img, base_cmap, base_alpha,
                     over_img, over_cmap, over_alpha):
    """ Taking two ResampledVolumeSlicers and corresponding colormaps,
    blend the two images into an RGBA byte array shaped like the resampled
    image of base_img.

    Parameters
    ----------
    base_img : ResampledVolumeSlicer
        The underlying image
    base_cmap : matplotlib.colors.LinearSegmentedColormap
        The mapping function to convert base_img's scalars to RGBA colors
    base_alpha : scalar or len-256 iterable (optional)
        the strength of this map when alpha blending;
        scalar in range [0,1], iterable should be ints from [0,255]
    over_img : ResampledVolumeSlicer
        The overlying image
    over_cmap : matplotlib.colors.LinearSegmentedColormap
        The mapping function to convert over_img's scalars to RGBA colors
    over_alpha : scalar or len-256 iterable (optional)
        the strength of this map when alpha blending;
        scalar in range [0,1], iterable should be ints from [0,255]

    Returns
    -------
    blended_arr : ndarray, dtype=uint8
        The blended RGBA array
    """
    base_bytes = normalize_and_map(base_img.image_arr, base_cmap,
                                   alpha=base_alpha)
    base_spacing = base_img.grid_spacing
    base_origin = base_img.coordmap.affine[:3,-1]
    over_bytes = normalize_and_map(over_img.image_arr, over_cmap,
                                   alpha=over_alpha)
    over_spacing = over_img.grid_spacing
    over_origin = over_img.coordmap.affine[:3,-1]
    resample_and_blend(base_bytes, base_spacing, base_origin,
                       over_bytes, over_spacing, over_origin)
    return base_bytes


def normalize_and_map(arr, cmap, alpha=1, norm_min=None, norm_max=None):
    """ Taking a scalar array, normalize the array to the range [0,1],
    and map the values through cmap into RGBA bytes

    Parameters
    ----------
    arr : ndarray
        array to map
    cmap : matplotlib.colors.LinearSegmentedColormap
        the mapping function
    alpha : scalar or len-256 iterable (optional)
        the strength of this map when alpha blending;
        scalar in range [0,1], iterable should be ints from [0,255]
    norm_min : scalar (optional)
        the value to rescale to zero (possibly clipping the transformed array)
    norm_max : scalar (optional)
        the value to rescale to one (possibly clipping the transformed array)
    """
    norm = mpl.colors.Normalize(vmin=norm_min, vmax=norm_max)
    scaled = norm(arr)
    # XYZ: this is quite a hack to get things going for now
    if hasattr(alpha, '__iter__'):
        cmap = mpl.colors.LinearSegmentedColormap('new', cmap._segmentdata,
                                                  N=256)
        cmap._init()
        lut = cmap._lut*255
        lut[:256,-1] = alpha
        lut = lut.astype(np.uint8)
        if np.ma.getmask(scaled) is not np.ma.nomask:
            xa = scaled.filled(fill_value=0)
        else:
            xa = scaled.copy()
        np.putmask(xa, xa==1.0, .999999)
        np.clip(xa*256, -1, 256, out=xa)
        bytes = lut.take(xa.astype('i'), axis=0, mode='clip')
    else:
        bytes = cmap(scaled, alpha=alpha, bytes=True)
    return bytes

def quick_min_max_norm(arr):
    an = arr.copy()
    an -= np.nanmin(an)
    an /= np.nanmax(an)
    return an

def resample_and_blend(base_arr, base_dr, base_r0,
                       over_arr, over_dr, over_r0):
    """ Taking two specs for VTKImageData-like data (array, spacing, origin),
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
        The origin coordinates of the base array, for each grid dimension
    over_arr : ndarray, dtype=uint8, len(shape) > 2
        RGBA byte values for the overlying array
    over_dr : iterable
        The grid spacing of the overlying array, for each grid dimension
    over_r0 : iterable
        The origin coordinates of the overlying array, for each grid dimension

    Returns
    -------
    Blends the two arrays into the base_arr
    """
    # This C++ code will resample over_arr onto base_arr with nearest neighbor
    # sampling. Both arrays expected to be shaped (nx, ny, nz, 4), and
    # contain unsigned 8bit integer RGBA values

    from scipy import weave
    # code requires ['base_arr', 'over_arr', 'b_dr', 'b_r0', 'o_dr', 'o_r0']
    resample_and_blend_src = """

using namespace blitz;
int i,j,k, ii, jj, kk;
int bnx, bny, bnz, onx, ony, onz;
unsigned bpr, bpg, bpb, bpa;
unsigned opr, opg, opb, opa;
double x, y, z;

bnx = base_arr.shape()[0]; bny = base_arr.shape()[1]; bnz = base_arr.shape()[2];
onx = over_arr.shape()[0]; ony = over_arr.shape()[1]; onz = over_arr.shape()[2];

for(i=0; i<bnx; i++) {
for(j=0; j<bny; j++) {
for(k=0; k<bnz; k++) {
x = i*b_dr(0) + b_r0(0); y = j*b_dr(1) + b_r0(1); z = k*b_dr(2) + b_r0(2);
ii = (int) ( (x - o_r0(0)) / o_dr(0) );
jj = (int) ( (y - o_r0(1)) / o_dr(1) );
kk = (int) ( (z - o_r0(2)) / o_dr(2) );
if( ii>=0 && ii<onx && jj>=0 && jj<ony && kk>=0 && kk<onz ) {
// do pixel blending into base_arr
bpr = (unsigned) base_arr(i,j,k,0); bpg = (unsigned) base_arr(i,j,k,1);
bpb = (unsigned) base_arr(i,j,k,2); bpa = (unsigned) base_arr(i,j,k,3);
opr = (unsigned) over_arr(ii,jj,kk,0); opg = (unsigned) over_arr(ii,jj,kk,1);
opb = (unsigned) over_arr(ii,jj,kk,2); opa = (unsigned) over_arr(ii,jj,kk,3);
base_arr(i,j,k,0) = (((opr-bpr)*opa + (bpr<<8))>>8);
base_arr(i,j,k,1) = (((opg-bpg)*opa + (bpg<<8))>>8);
base_arr(i,j,k,2) = (((opb-bpb)*opa + (bpb<<8))>>8);
//base_arr(i,j,k,3) = ((opa+bpa) - ((opa*bpa + 255) >> 8));
}
//else {
//std::cout<<"skipping voxel "<<i<<","<<j<<","<<k<<std::endl;
//}
}
}
}

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
    bshape = list(base_arr.shape)
    base_dr = list(base_dr); base_r0 = list(base_r0)
    while len(bshape) < 4:
        bshape.insert(0,1)
        base_dr.insert(0,1)
        base_r0.insert(0,0)
    base_arr.shape = tuple(bshape)
    b_dr = np.array(base_dr)
    b_r0 = np.array(base_r0)
        
    oshape = list(over_arr.shape)
    over_dr = list(over_dr); over_r0 = list(over_r0)
    while len(oshape) < 4:
        oshape.insert(0,1)
        over_dr.insert(0,1)
        over_r0.insert(0,0)
    over_arr.shape = tuple(oshape)
    o_dr = np.array(over_dr)
    o_r0 = np.array(over_r0)
        
    arg_list = ['base_arr', 'over_arr', 'b_dr', 'b_r0', 'o_dr', 'o_r0']
    weave.inline(resample_and_blend_src, arg_list,
                 type_converters=weave.converters.blitz)
    base_arr.shape = tuple( filter(lambda x: x>1, bshape) )
    over_arr.shape = tuple( filter(lambda x: x>1, oshape) )
