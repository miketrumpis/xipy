# NumPy / Scipy
import numpy as np
from scipy import ndimage

# Matplotlib normalization
from matplotlib import colors

# NIPY
from nipy.core import api as ni_api

# XIPY
from xipy.slicing import SAG, COR, AXI, transverse_plane_lookup, enumerated_axes
from xipy.external.interpolation import ImageInterpolator
import xipy.volume_utils as vu
import xipy.vis.color_mapping as cm


class VolumeSlicerInterface(object):
    """
    Interface only class!

    This class defines an interface for all different VolumeSlicerInterface
    types,
    specifically the cut_image() usage.
    """
    def __init__(self, image, bbox=None, mask=False,
                 grid_spacing=None, fliplr=False):
        """
        Creates a new VolumeSlicerInterface
        
        Parameters
        ----------
        image : a NIPY Image
            The image to slice
        bbox : iterable (optional)
            The {x,y,z} limits of the enrounding volume box. If None, then
            slice planes in the natural box of the image. This argument
            is useful for overlaying an image onto another image's volume box
        mask : bool or ndarray (optional)
            A binary mask, with same shape as image, with unmasked points
            marked as True (opposite of MaskedArray convention)
        grid_spacing : iterable (optional)
            New grid spacing for the sliced planes. If None, then the
            natural voxel spacing is used.
        fliplr : bool (optional)
            Set True if the transform from voxel indices to world coordinates
            maps to a left-handed space, (Radiological convention)
        """
        raise NotImplementedError('This interface class is not for real use')


    def _cut_plane(self, ax, coord, **interp_kw):
        """
        For a given axis in {SAG, COR, AXI}, make a plane cut in the
        volume at the coordinate value.

        Parameters
        ----------
        ax : int
            axis label in {SAG, COR, AXI} (defined in xipy.slicing)
        coord : float
            coordinate value along this axis
            
        Returns
        _______
        plane : ndarray
            The transverse plane
        """
        pass
    
    def cut_image(self, loc, axes=(SAG, COR, AXI), oriented=True, **interp_kw):
        """
        Return len(axes) planes, which are cut along the axes specified.
        By default, the planes will be rotated if necessary to be
        aligned in the traditional orientation:
        
        axial slice: (left-right by posterior-anterior)
        coronal slice: (left-right by inferior-superior)
        sagittal slice: (posterior-anterior by inferior-superior)

        To return planes as-they-are-sliced in the array,
        set oriented=False

        Parameters
        ----------
        loc : iterable, len-3
            The coordinates of the cut location
        axes : iterable, len-1, 2, or 3
            The returned planes will be those normal to these axes (by default
            all three SAG, COR, AXI axes -- also may be specified by name
            in terms of 'SAG', 'COR', 'AXI', or 'x', 'y', 'z')
        aligned : bool
            Whether to return the planes aligned to the canonical orientations
        interp_kw : dict
            Keyword args for the interpolating machinery
            (ie, ndimage.map_coordinates keyword args--not implemented
            in all subclasses)

        Returns
        _______
        len(axes) planes
        """
        enum_axes = enumerated_axes(axes)
        axes = [ni_api.ras_output_coordnames[ax] for ax in enum_axes]
        indices = self.coordmap.inverse()(loc)
        planes = [self._cut_plane(ax, indices, oriented=oriented, **interp_kw)
                  for ax in axes]
        return planes

    def update_mask(self, mask, positive_mask=True):
        """
        Reset the mask of the raw image data.

        Parameters
        ----------
        mask : ndarray
            The new mask
        positive_mask : bool
            Indicates whether this is a positive (True=unmasked) or
            negative (True=masked) style mask
       """
        pass

    @classmethod
    def from_blobs(klass, scalar_map, voxel_coordinates, coordmap, **kwargs):
        arr_indices = np.round(coordmap.inverse()(voxel_coordinates)).astype('i')
##         arr_indices = coordmap.inverse(voxel_coordinates).astype('i')
        arr = vu.signal_array_to_masked_vol(scalar_map, arr_indices)
        image = ni_api.Image(arr.data, coordmap)
        return klass(image, mask=np.logical_not(arr.mask), **kwargs)
                     


class SampledVolumeSlicer(VolumeSlicerInterface):
    """
    This object cuts up an image along the axes defined by its
    CoordinateMap target space. The SampledVolumeSlicer provides slices
    through an image such that the cut planes extend across the three
    {x,y,z} planes of the target space. Each plane is sampled from the
    original image voxel array by spline interpolation.
    """

    def __init__(self, image, bbox=None, mask=False,
                 grid_spacing=None, fliplr=False, interpolation_order=3):
        """
        Creates a new SampledVolumeSlicer
        
        Parameters
        ----------
        image : a NIPY Image
            The image to slice
        bbox : iterable (optional)
            The {x,y,z} limits of the enrounding volume box. If None, then
            slices planes in the natural box of the image. This argument
            is useful for overlaying an image onto another image's volume box
        mask : bool or ndarray (optional)
            A binary mask, with same shape as image, with unmasked points
            marked as True (opposite of MaskedArray convention)
        grid_spacing : iterable (optional)
            New grid spacing for the sliced planes. If None, then the
            natural voxel spacing is used.
        fliplr : bool (optional)
            Set True if the transform from voxel indices to world coordinates
            maps to a left-handed space, (Radiological convention)
        """
        
        image = vu.fix_analyze_image(image, fliplr=fliplr)
        xyz_image = ni_api.Image(
            np.asarray(image),
            image.coordmap.reordered_range(ni_api.ras_output_coordnames)
            )
##                                  reorder_output(image.coordmap, 'xyz'))
        self.coordmap = xyz_image.coordmap
        self.raw_image = xyz_image
        nvox = np.product(self.raw_image.shape)
        # if the volume array of the map is more than 100mb in memory,
        # better use a memmap
        self._use_mmap = nvox*8 > 100e6
        self.interpolator = ImageInterpolator(xyz_image,
                                              order=interpolation_order,
                                              use_mmap=self._use_mmap)
##         if mask is True:
##             mask = compute_mask(np.asarray(self.raw_image), cc=0, m=.1, M=.999)
        if type(mask) is np.ndarray:
            self.update_mask(mask, positive_mask=True)
        else:
            self._masking = False
            self.raw_mask = None
            
        if grid_spacing is None:
            self.grid_spacing = list(vu.voxel_size(image.affine))
        else:
            self.grid_spacing = grid_spacing
        if bbox is None:
            self._nominal_bbox = vu.world_limits(xyz_image)
        else:
            self._nominal_bbox = bbox

        # defines {x,y,z}shape and {x,y,z}grid
        self._define_grids()

    def _define_grids(self):
        dx, dy, dz = self.grid_spacing
        xbox, ybox, zbox = map( lambda x: np.array(x).reshape(2,2),
                                vu.limits_to_extents(self._nominal_bbox) )
        
        # xbox is [ <ylims> , <zlims> ], so xgrid is [ypts.T, zpts.T]
        xr = np.diff(xbox)[:,0]
        xspacing = np.array([dy, dz])
        self.xshape = (xr / xspacing).astype('i')
        aff = ni_api.AffineTransform.from_start_step('ij', 'xy',
                                                     xbox[:,0], xspacing)
        self.xgrid = ni_api.ArrayCoordMap(aff, self.xshape).values

        # ybox is [ <xlims>, <zlims> ], so ygrid is [xpts.T, zpts.T]
        yr = np.diff(ybox)[:,0]
        yspacing = np.array([dx, dz])
        self.yshape = (yr / yspacing).astype('i')
        aff = ni_api.AffineTransform.from_start_step('ij', 'xy',
                                                     ybox[:,0], yspacing)
        self.ygrid = ni_api.ArrayCoordMap(aff, self.yshape).values

        # zbox is [ <xlims>, <ylims> ], so zgrid is [xpts.T, ypts.T]
        zr = np.diff(zbox)[:,0]
        zspacing = np.array([dx, dy])
        self.zshape = (zr / zspacing).astype('i')
        aff = ni_api.AffineTransform.from_start_step('ij', 'xy',
                                                     zbox[:,0], zspacing)
        self.zgrid = ni_api.ArrayCoordMap(aff, self.zshape).values

        bb_min = [b[0] for b in self._nominal_bbox]
        lx = self.yshape[0]
        ly = self.xshape[0]
        lz = self.yshape[1]
        bb_max = [bb_min[0] + dx*lx, bb_min[1] + dy*ly, bb_min[2] + dz*lz]
        self.bbox = zip(bb_min, bb_max)
        
    def update_mask(self, mask, positive_mask=True):
        cmap = self.coordmap
        if not positive_mask:
            # IE, if this is a MaskedArray type mask
            mask = np.logical_not(mask)
##         fmask = np.array([ndimage.binary_fill_holes(m) for m in mask], 'd')
        self.m_interpolator = ImageInterpolator(ni_api.Image(mask, cmap),
                                                order=3,
                                                use_mmap=self._use_mmap)
        self.raw_mask = mask
        self._masking = True

    def _update_grid_spacing(self, grid_spacing):
        self.grid_spacing = grid_spacing
        self._define_grids()

##     def update_mask_crit(self, crit, thresh):
##         self._masking = True
##         self._mcrit = crit
##         self._thresh = thresh

    def _closest_grid_pt(self, coord, ax):
        """For a given coordinate value on an axis, find the closest
        fixed grid point as defined on the complementary grid planes.
        This is to ensure consistency at the intersection of
        interpolated planes
        """
        ax_min = np.asarray(self.bbox)[ax,0]
        ax_step = self.grid_spacing[ax]
        # ax_min + n*ax_step = coord
        new_coord = round ( (coord-ax_min) / ax_step ) * ax_step + ax_min
        return new_coord
        
    def _cut_plane(self, ax, coord, **interp_kw):
        """
        For a given axis in {SAG, COR, AXI}, make a plane cut in the
        volume at the coordinate value.

        Parameters
        ----------
        ax : int
            axis label in {SAG, COR, AXI} (defined in xipy.slicing)
        coord : float
            coordinate value along this axis
        interp_kw : dict
            Keyword args for the interpolating machinery
            (ie, ndimage.map_coordinates keyword args)
            
        Returns
        _______
        plane : ndarray
            The transverse plane sampled at the grid points and fixed axis
            coordinate for the given args
        """
    
        # a little hokey
        grid_lookup = {SAG: ('xshape', 'xgrid'),
                       COR: ('yshape', 'ygrid'),
                       AXI: ('zshape', 'zgrid')}
        sname, gname= grid_lookup[ax]
        ii, jj = transverse_plane_lookup(ax)
        grid = getattr(self, gname)
        shape = getattr(self, sname)
        coords = np.empty((3, grid.shape[0]), 'd')
        coords[ax] = self._closest_grid_pt(coord, ax)
        coords[ii] = grid[:,0]; coords[jj] = grid[:,1]
        pln = self.interpolator.evaluate(coords, **interp_kw).reshape(shape)
        if self._masking:
            m_pln = self.m_interpolator.evaluate(coords, mode='constant',
                                                 cval=-10).reshape(shape)
            return np.ma.masked_where(m_pln < 0.5, pln)
        return pln

    def update_target_space(self, coreg_image):
        raise NotImplementedError('not sure how to do this yet')

class ResampledVolumeSlicer(VolumeSlicerInterface):
    """
    This object cuts up an image along the axes defined by its
    CoordinateMap target space. The ResampledVolumeSlicer provides slices
    through an image such that the cut planes extend across the three
    {x,y,z} planes of the target space. Each plane is cut from a fully
    resampled image array.
    """
    
    def __init__(self, image, bbox=None, mask=False,
                 grid_spacing=None, fliplr=False, **interp_kws):
        """
        Creates a new ResampledVolumeSlicer
        
        Parameters
        ----------
        image : a NIPY Image
            The image to slice
        bbox : iterable (optional)
            The {x,y,z} limits of the enrounding volume box. If None, then
            slices planes in the natural box of the image. This argument
            is useful for overlaying an image onto another image's volume box
        mask : bool or ndarray (optional)
            A binary mask, with same shape as image, with unmasked points
            marked as True (opposite of MaskedArray convention)
        grid_spacing : iterable (optional)
            New grid spacing for the sliced planes. If None, then the
            natural voxel spacing is used.
        fliplr : bool (optional)
            Set True if the transform from voxel indices to world coordinates
            maps to a left-handed space, (Radiological convention)
        interp_kws : dict
            Keyword args for the interpolating machinery.. eg:
            order -- spline order
            mode --  Points outside the boundaries of the input are filled
                     according to the given mode ('constant', 'nearest',
                     'reflect' or 'wrap'). Default is 'constant'.
            cval -- fill value if mode is 'constant'
            
        """

        image = vu.fix_analyze_image(image, fliplr=fliplr)
        # XYZ: NEED TO BREAK API HERE FOR MASKED ARRAY
        xyz_image = ni_api.Image(
            image._data,
            image.coordmap.reordered_range(ni_api.ras_output_coordnames)
            )

        if bbox is None:
            bbox = vu.world_limits(xyz_image)
        self.grid_spacing = vu.voxel_size(xyz_image.affine) \
                            if grid_spacing is None else grid_spacing

        # only resample if the image actually need a warping/rotation..
        # if the requested bounding box is different than the natural
        # bounding box, then don't resample but be smart when indexing
        if not vu.is_spatially_aligned(image.coordmap):
            print 'resampling entire Image volume'
            world_image = vu.resample_to_world_grid(
                image, grid_spacing=self.grid_spacing,
                **interp_kws
                )
        else:
            world_image = xyz_image

        self.coordmap = world_image.coordmap
        self.image_arr = np.asarray(world_image)

        # take down the final bounding box; this will define the
        # field of the overlay plot
        bb_min = world_image.affine[:3,-1]
        bb_max = [b0 + l*dv for b0, l, dv in zip(bb_min,
                                                 world_image.shape,
                                                 self.grid_spacing)]
        self.bbox = zip(bb_min, bb_max)
        
##         if type(mask) is np.ndarray:
##             # sets self.raw_mask and self._mask and self._masking=True,
##             # also converts self.image_arr to a MaskedArray
##             self.update_mask(mask)
##         else:
##             self._masking = False
##             self.raw_mask = None

        # now find the logical axis to array axis mapping
        T = np.abs(self.coordmap.affine[:3,:3])
        logical_to_array = [r.argmax() for r in T]
        self._ax_lookup = dict( zip(range(3), logical_to_array) )
        o_coords = self.coordmap.function_range.coord_names
        self._ax_lookup.update( dict( zip(o_coords, logical_to_array) ) )
        self._ax_lookup.update( dict( zip(['SAG', 'COR', 'AXI'],
                                          logical_to_array) ) )
    
        w_shape = world_image.shape
        # these planes are shaped as if the image_arr were
        # sliced along a given axis
        self.null_planes = [np.ma.masked_all((w_shape[0], w_shape[1]),'B'),
                            np.ma.masked_all((w_shape[0], w_shape[2]),'B'),
                            np.ma.masked_all((w_shape[1], w_shape[2]),'B')]
    
    def update_mask(self, mask, positive_mask=True):
        # XYZ: CURRENTLY OBLITERATES OLD MASK! IS THIS DESIRABLE?
        assert mask.shape == self.raw_image.shape, \
                          'mask shape does not match image shape'
        self._masking = True
        if positive_mask:
            self.raw_mask = mask.astype(np.bool)
        else:
            self.raw_mask = np.logical_not(mask.astype(np.bool))
        aff = self.coordmap.affine
        if not np.allclose(aff.diagonal()[:3], self.grid_spacing):
            m_image = ni_api.Image(self.raw_mask.astype('d'), self.coordmap)
            resamp_mask = vu.resample_to_world_grid(
                m_image, bbox=self.bbox, grid_spacing=self.grid_spacing
                )
            # now world_mask is a negative mask
            self._mask = np.asarray(resamp_mask) < 0.5
        else:
            self._mask = np.logical_not(self.raw_mask)

        print 'new unmasked pts:', np.logical_not(self._mask).sum()
        # XYZ: IF THE DATA COMES INTO THIS CLASS AS A MASKED ARRAY,
        # THE ORIGINAL FILL VALUE GETS LOST HERE
        self.image_arr = np.ma.masked_array(np.ma.getdata(self.image_arr),
                                            mask=self._mask)
        
        
    def _cut_plane(self, ax, indices, oriented=True):
        """
        For a given axis name, find the points on the transverse grid
        and make a cut at the given coord

        Parameters
        ----------
        ax : str
            logical axis name
        coord : len-3 iterable
            index coordinates
        oriented : bool, optional
            return the plane oriented in the canonical layout
        Returns
        _______
        plane : ndarray
            The transverse plane cut along the given axis and coordinate
        """
        arr_ax = self._ax_lookup[ax]
        idx = int(indices[arr_ax])
        
        dim_size = self.image_arr.shape[arr_ax]
        if idx < 0 or idx >= dim_size:
            pln = self.null_planes[arr_ax]
        else:
            slicer = [slice(None)]*3
            slicer[arr_ax] = idx
            pln = self.image_arr[tuple(slicer)]

        if oriented:
            ras = list(ni_api.ras_output_coordnames)
            output_order = {
                ras[0] : (ras[2], ras[1]),
                ras[1] : (ras[2], ras[0]),
                ras[2] : (ras[1], ras[0])
                }[ax]
            aff=vu.drop_io_dim(self.coordmap, ax).reordered_range(output_order)
            # if the little aff is diagonal, let it go
            # otherwise transpose the plane
            if np.abs(aff.affine[0,0]) < np.abs(aff.affine[0,1]):
                # only transpose the plane axes (0 and 1), in
                # case this plane has vector information in the last dimension
                axes = np.arange(len(pln.shape))
                axes[0] = 1; axes[1] = 0
                return pln.transpose(*axes)
        return pln

class ResampledIndexVolumeSlicer(ResampledVolumeSlicer):
    """
    This class creates a resampled volume of indices into a color LUT,
    and otherwise operates identically as a ResampledVolumeSlicer.
    """

    def __init__(self, image, bbox=None, norm=None,
                 grid_spacing=None, fliplr=False):
        """
        Creates a new ResampledVolumeSlicer
        
        Parameters
        ----------
        image : a NIPY Image
            The image to slice
        bbox : iterable (optional)
            The {x,y,z} limits of the enrounding volume box. If None, then
            slices planes in the natural box of the image. This argument
            is useful for overlaying an image onto another image's volume box
        norm : (black-pt, white-pt) pair or mpl.colors.Normalize instance
            Limits for normalizing the scalar values. If image contains
        grid_spacing : iterable (optional)
            New grid spacing for the sliced planes. If None, then the
            natural voxel spacing is used.
        fliplr : bool (optional)
            Set True if the transform from voxel indices to world coordinates
            maps to a left-handed space, (Radiological convention)
        """
        image = vu.fix_analyze_image(image, fliplr=fliplr)
        
        # XYZ: NEED TO BREAK API HERE FOR MASKED ARRAY
        vol_data = np.ma.masked_array(image._data)
        if norm is not False:
            # normalize the image data to the range [0,1]
            if norm is None or norm==(0, 0):
                norm = colors.Normalize()
            elif type(norm) in (list, tuple):
                norm = colors.Normalize(*norm)
            elif type(norm) is not colors.Normalize:
                raise ValueError('Could not parse normalization parameter')
            compressed = norm(vol_data)
            # map to indices
            raw_idx = cm.MixedAlphaColormap.lut_indices(compressed)
        else:
            raw_idx = np.asarray(image)

        idx_image = ni_api.Image(raw_idx, image.coordmap)
        # Resample to a diagonal affine with grid spacing as given.
        # Fill in boundary voxels with "i_bad", so they are hidden
        # in the color mapping
        bad_idx = cm.MixedAlphaColormap.i_bad
        ResampledVolumeSlicer.__init__(self, idx_image, bbox=bbox,
                                       grid_spacing=grid_spacing,
                                       order=0, cval=bad_idx)
        
    def update_mask(self, mask, positive_mask=True):
        raise NotImplementedError('no updating masks in index mapped images')

            
        
