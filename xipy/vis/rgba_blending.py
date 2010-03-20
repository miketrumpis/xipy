import numpy as np
import matplotlib as mpl
from _blend_pix import resample_and_blend

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
    base_origin = np.array(base_img.bbox)[:,0]
    over_bytes = normalize_and_map(over_img.image_arr, over_cmap,
                                   alpha=over_alpha)
    over_spacing = over_img.grid_spacing
    over_origin = np.array(over_img.bbox)[:,0]
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

