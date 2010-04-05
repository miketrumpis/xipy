import numpy as np
import matplotlib as mpl
from _blend_pix import resample_and_blend, resize_lookup_array
import _blend_pix
from time import time

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
    cmap : xipy.vis.color_mapping.MixedAlphaColormap
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
    bytes = cmap(scaled, alpha=alpha, bytes=True)
    return bytes

def quick_min_max_norm(arr):
    an = arr.copy()
    an -= np.nanmin(an)
    an /= np.nanmax(an)
    return an

import enthought.traits.ui.api as ui_api
import enthought.traits.api as t_ui
import xipy.vis.color_mapping as cm
from matplotlib.colors import Normalize
import xipy.volume_utils as vu
import nipy.core.api as ni_api

def blend_helper(a1, a2):
    xyz_shape = a1.shape[:3]
    npts = np.prod(xyz_shape)
    a1.shape = (npts, 4)
    a2.shape = (npts, 4)
    _blend_pix.blend_same_size_arrays(a1, a2)
    a1.shape = xyz_shape + (4,)
    a2.shape = xyz_shape + (4,)

class BlendedImage(t_ui.HasTraits):

    main_spline_order = t_ui.Range(low=0,high=5,value=0)
    over_spline_order = t_ui.Range(low=0,high=5,value=0)
    transpose_inputs = t_ui.Bool(True)

    # the scalar images
    main = t_ui.Instance(ni_api.Image) #ResampledVolumeSlicer)
    over = t_ui.Instance(ni_api.Image) #ResampledVolumeSlicer)

    img_spacing = t_ui.Property(depends_on='main')
    img_origin = t_ui.Property(depends_on='main')

    # the LUT index images
    _main_idx = t_ui.Array()
    _over_idx = t_ui.Array()

    # the RGBA byte arrays
    main_rgba = t_ui.Array(dtype='B', comparison_mode=t_ui.NO_COMPARE) #t_ui.Instance(t_ui.Array, dtype='B')
    over_rgba = t_ui.Array(dtype='B', comparison_mode=t_ui.NO_COMPARE) #t_ui.Instance(t_ui.Array, dtype='B')
    blended_rgba = t_ui.Property(depends_on='main_rgba, over_rgba') #t_ui.Array(dtype='B') #t_ui.Instance(t_ui.Array, dtype='B')

    # color mapping properties
    main_cmap = t_ui.Instance(cm.MixedAlphaColormap)
    over_cmap = t_ui.Instance(cm.MixedAlphaColormap)    

    main_norm = t_ui.Tuple((0.0, 0.0))
    over_norm = t_ui.Tuple((0.0, 0.0))

    main_alpha = t_ui.Any # can be a float or array??
    over_alpha = t_ui.Any 

    def __init__(self, **traits):
        # for now, main and over are ResampledVolumeSlicer types
        t_ui.HasTraits.__init__(self, **traits)
        if not self.main_cmap:
            self.set(main_cmap=cm.gray, trait_change_notify=False)
        if not self.over_cmap:
            self.set(over_cmap=cm.jet, trait_change_notify=False)
        if self.main_alpha is None:
            self.set(main_alpha=1.0, trait_change_notify=False)
        if self.over_alpha is None:
            self.set(over_alpha=1.0, trait_change_notify=False)
    @t_ui.cached_property
    def _get_img_spacing(self):
        spacing = vu.voxel_size(self.main.affine)
        return spacing[::-1] if not self.transpose_inputs else spacing
    @t_ui.cached_property
    def _get_img_origin(self):
        origin = np.array(vu.world_limits(self.main))[:,0]
        return origin[::-1] if not self.transpose_inputs else origin

    def _check_alpha(self, alpha):
        if mpl.cbook.iterable(alpha):
            return np.clip(alpha, 0, 1)
        # assuming both cmaps have the same # of colors!!
        return np.ones(self.main_cmap.N)*max(0.0, min(alpha, 1.0))

##     @t_ui.on_trait_change('main_rgba, over_rgba')
##     def _reblend_bytes(self):
    @t_ui.cached_property
    def _get_blended_rgba(self):
##         if self.blended_rgba.shape == (0,):
##             self.blended_rgba = self.main_rgba.copy()
##         else:
##             self.blended_rgba[:] = self.main_rgba
        blended_rgba = self.main_rgba.copy()
        if self.over:
            blend_helper(blended_rgba, self.over_rgba)
        return blended_rgba

    @t_ui.on_trait_change('main, main_spline_order, main_norm')
    def _update_mbytes(self, name, new):
        if not self.main:
            return
        if self.main_norm != (0., 0.):
            n = mpl.colors.Normalize(*self.main_norm)
        else:
            n = mpl.colors.Normalize()
        compressed = n(self.main._data)
        raw_idx = self.main_cmap.lut_indices(compressed)
        main_idx_image = vu.resample_to_world_grid(
            ni_api.Image(raw_idx, self.main.coordmap),
            order=self.main_spline_order
            )
        # would be nice to transpose before resampling
        if self.transpose_inputs:
            self._main_idx = np.asarray(main_idx_image).transpose().copy()
        else:
            self._main_idx = np.asarray(main_idx_image)
        self.main_rgba = self.main_cmap.fast_lookup(
            self._main_idx, alpha=self.main_alpha, bytes=True
            )
        # if the main grid changed, need to update obytes
        if name=='main':
            self._update_obytes('foo', 'foo')
        
    @t_ui.on_trait_change('over, over_spline_order, over_norm')
    def _update_obytes(self, changed, new):
        print 'updating obytes from changed:', changed
        if not self.over:
            return
        if self.over_norm != (0., 0.):
            n = mpl.colors.Normalize(*self.over_norm)
        else:
            n = mpl.colors.Normalize()
        compressed = n(self.over._data)
        raw_idx = self.over_cmap.lut_indices(compressed)
        over_idx_image = vu.resample_to_world_grid(
            ni_api.Image(raw_idx, self.over.coordmap),
            order=self.over_spline_order
            )
        if self.transpose_inputs:
            temp_over_idx = np.asarray(over_idx_image).transpose().copy()
        else:
            temp_over_idx = np.asarray(over_idx_image)
        if not self.main:
            self._over_idx = temp_over_idx
        else:
            # still need to quickly upsample this volume into the
            # main volume grid
            over_r0 = np.array(vu.world_limits(over_idx_image))[:,0]
            over_dr = vu.voxel_size(over_idx_image.affine)
            main_r0 = np.array(vu.world_limits(self.main))[:,0]
            main_dr = vu.voxel_size(self.main.affine)
            ibad = self.over_cmap._i_bad
            if self.transpose_inputs:
                self._over_idx = resize_lookup_array(
                    self.main.shape, ibad,
                    temp_over_idx,
                    over_dr[::-1], over_r0[::-1],
                    main_dr[::-1], main_r0[::-1])
            else:
                self._over_idx = resize_lookup_array(
                    self.main.shape, ibad,
                    temp_over_idx,
                    over_dr, over_r0,
                    main_dr, main_r0)
                
        self.over_rgba = self.over_cmap.fast_lookup(
            self._over_idx, alpha=self.over_alpha, bytes=True
            )
        return
        

    @t_ui.on_trait_change('main_cmap')
    def _remap_main(self):
        if not self.main:
            return
        self.main_rgba[:] = self.main_cmap.fast_lookup(
            self._main_idx, alpha=self.main_alpha, bytes=True
            )
        # have to do this explicitly to set off trait notification
        self.main_rgba = self.main_rgba

    @t_ui.on_trait_change('over_cmap')
    def _remap_over(self):
        print 'remapping over bytes', 
        if not self.over:
            print ' but no overlay'
            return
        print ''
        self.over_rgba[:] = self.over_cmap.fast_lookup(
            self._over_idx, alpha=self.over_alpha, bytes=True
            )
        # have to do this explicitly to set off trait notification
        self.over_rgba = self.over_rgba


    @t_ui.on_trait_change('main_alpha')
    def _fast_remap_main_alpha(self):
        if not self.main:
            return
        main_alpha = self._check_alpha(self.main_alpha)
        a_under, a_over, a_bad = self.main_cmap._lut[-3:,-1]
        alpha_lut = np.r_[main_alpha*255, a_under, a_over, a_bad]
        alpha_lut.take(self._main_idx, axis=0,
                       mode='clip', out=self.main_rgba[...,3])
        print 'looked up new main alpha chan'
        self.trait_setq(main_alpha=main_alpha)
        # have to do this explicitly to set off trait notification
        self.main_rgba = self.main_rgba

    @t_ui.on_trait_change('over_alpha')
    def _fast_remap_over_alpha(self):
        print 'remapping over alpha bytes',
        if not self.over:
            print 'but no overlay'
            return
        print ''
        over_alpha = self._check_alpha(self.over_alpha)
        a_under, a_over, a_bad = self.over_cmap._lut[-3:,-1]
        alpha_lut = np.r_[over_alpha*255, a_under, a_over, a_bad]
        alpha_lut.take(self._over_idx, axis=0,
                       mode='clip', out=self.over_rgba[...,3])
        print 'looked up new over alpha chan'
        self.trait_setq(over_alpha=over_alpha)
        # have to do this explicitly to set off trait notification
        self.over_rgba = self.over_rgba

##     @t_ui.on_trait_change('over_rgba, main_rgba, blended_rgba')
##     def _test_trait_notify(self, name, new):
##         print name, 'got changed in BlendedImage'
    
    # handle norm later.. I'm thinking this can be accomplished with a
    # sort of transfer function from integer indices to indices
