"""This module has some local hackups of Mayavi objects"""
import numpy as np

from enthought.tvtk.api import tvtk
from enthought.tvtk import array_handler
import enthought.mayavi.sources.api as src_api
from enthought.traits.api import Trait, Instance, CInt, Range, Enum
from enthought.mayavi.modules.image_plane_widget import ImagePlaneWidget
from enthought.mayavi.tools.modules import DataModuleFactory
from enthought.mayavi.tools.pipe_base import make_function

import time
def time_wrap(fcall, ldict, gdict=None):
    if gdict is None:
        gdict = globals()
    t0 = time.time()
    exec fcall in gdict, ldict
    t = time.time()
    print '%s: %1.3f sec'%(fcall, (t-t0))

# -- A New RGBA-enabled ArraySource ------------------------------------------
def _check_scalar_array(obj, name, value):
    """Validates a possibly multi-component scalar array.
    """
    if value is None:
        return None
    arr = np.asarray(value)
    assert arr.dtype.char == 'B', "Scalar array dtype must be unsigned 8bit int"
    assert len(arr.shape) in [3,4], "Scalar array must be {3,4} dimensional"
    assert arr.shape[-1] == 4, \
               "Component dimension for scalar array must be length-4"
    vd = obj.vector_data
    if vd is not None:
        arr_dims = arr.shape[:min(arr.ndim, 3)]
        assert vd.shape[:-1] == arr_dims, \
               "Scalar array must match already set vector data.\n"\
               "vector_data.shape = %s, given array shape = %s"%(vd.shape,
                                                                 arr.shape)
    return arr

_check_scalar_array.info = 'a 2-, 3-, or 4-D numpy array'

class ArraySourceRGBA(src_api.ArraySource):
    """This is a version of ArraySource that allows for assignment
    of RGBA component data -- IE,
    data.shape must be (nx, ny, [nz], 4)
    data.dtype must be np.uint8
    """
    scalar_data = Trait(None, _check_scalar_array, rich_compare=True)

    def _scalar_data_changed(self, data):
        import numpy
        img_data = self.image_data
        if data is None:
            img_data.point_data.scalars = None
            self.data_changed = True
            return
        is_rgba_bytes = (data.dtype.char=='B' and data.shape[-1]==4)
        dims = list(data.shape[:-1]) if is_rgba_bytes else list(data.shape)
        if len(dims) == 2:
            dims.append(1)
      
        img_data.origin = tuple(self.origin)
        img_data.dimensions = tuple(dims)
        img_data.extent = 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1
        img_data.update_extent = 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1

        flat_shape = ( numpy.prod(dims), )
        if is_rgba_bytes:
            flat_shape += (4,)
        if self.transpose_input_array:
            if is_rgba_bytes:
                # keep the color components in the last dimension
                d = data.transpose(2,1,0,3).copy()
                d.shape = flat_shape
                img_data.point_data.scalars = d
            else:
                img_data.point_data.scalars = numpy.ravel(numpy.transpose(data))
        else:
            img_data.point_data.scalars = data.reshape(flat_shape)

        img_data.number_of_scalar_components = 4 if is_rgba_bytes else 1
        img_data.point_data.scalars.name = self.scalar_name
        # This is very important and if not done can lead to a segfault!
        typecode = data.dtype
        img_data.scalar_type = array_handler.get_vtk_array_type(typecode)
        img_data.update() # This sets up the extents correctly.
        img_data.update_traits()
        self.change_information_filter.update()

        # Now flush the mayavi pipeline.
        self.data_changed = True


    def update(self):
        """Call this function when you change the array data
        in-place."""
        d = self.image_data
        d.modified()
        pd = d.point_data
        if self.scalar_data is not None:
            pd.scalars.modified()
        if self.vector_data is not None:
            pd.vectors.modified()
##         self.data_changed = True

# -- An ImagePlaneWidget & Helpers that turns off the LUT --------------------
class ImagePlaneWidget_RGBA(ImagePlaneWidget):

    def update_pipeline(self):
        """Override this method so that it *updates* the tvtk pipeline
        when data upstream is known to have changed.

        This method is invoked (automatically) when any of the inputs
        sends a `pipeline_changed` event.
        """
        mod_mgr = self.module_manager
        if mod_mgr is None:
            return

        # Data is available, so set the input for the IPW.
        input = mod_mgr.source.outputs[0]
        if not (input.is_a('vtkStructuredPoints') \
                or input.is_a('vtkImageData')):
            msg = 'ImagePlaneWidget only supports structured points or '\
                  'image data.'
            error(msg)
            raise TypeError, msg
            
        self.ipw.input = input
        self.ipw.lookup_table = None
        self.ipw.color_map.lookup_table = None
##         # Set the LUT for the IPW.
##         if self.use_lookup_table:
##             self.ipw.lookup_table = mod_mgr.scalar_lut_manager.lut
        
        self.pipeline_changed = True

class ImagePlaneWidgetFactory_RGBA(DataModuleFactory):
    """ Applies the ImagePlaneWidget mayavi module to the given data
        source (Mayavi source, or VTK dataset). 
    """
    _target = Instance(ImagePlaneWidget_RGBA, ())

    slice_index = CInt(0, adapts='ipw.slice_index',
                        help="""The index along wich the
                                            image is sliced.""")

    plane_opacity = Range(0.0, 1.0, 1.0, adapts='ipw.plane_property.opacity',
                    desc="""the opacity of the plane actor.""")

    plane_orientation = Enum('x_axes', 'y_axes', 'z_axes',
                        adapts='ipw.plane_orientation',
                        desc="""the orientation of the plane""")

image_plane_widget_rgba = make_function(ImagePlaneWidgetFactory_RGBA)

# -- An ArraySource with "channels" ------------------------------------------
from xipy.colors.mayavi_tools import ArraySourceRGBA
from xipy.colors.rgba_blending import BlendedImages, quick_convert_rgba_to_vtk
class MasterSource(ArraySourceRGBA):
    """
    This class monitors a BlendedImages object and sets up image
    channels for the main image, over image, and blended image.

    It may have additional channels
    """

    # XXX: This source should have a signal saying when a channel may
    # disappear!!! EG, if blender.over gets set to None, and some module
    # is visualizing "over_colors", then there will be a big fat crash!!
    
    blender = t.Instance(BlendedImages)
    over_channel = 'over_colors'
    main_channel = 'main_colors'
    blended_channel = 'blended_colors'

    rgba_channels = t.Property
    all_channels = t.Property

    transpose_input_array = False

    @t.on_trait_change('transpose_input_array')
    def _ignore_transpose(self):
        self.trait_setq(transpose_input_array=False)

    @t.on_trait_change('blender')
    def _check_vtk_order(self):
        if not self.blender.vtk_order:
            raise ValueError('BlendedImages instance must be in VTK order')

    def _get_all_channels(self):
        pdata = self.image_data.point_data
        names = [pdata.get_array_name(n)
                 for n in xrange(pdata.number_of_arrays)]
        return names

    def _get_rgba_channels(self):
        primary_channels = (self.over_channel,
                            self.main_channel,
                            self.blended_channel)
        names = self.all_channels
        return [n for n in names if n in primary_channels]

    ## The convention will be to have main_rgba be the primary array
    ## in scalar_data. If main_rgba isn't present, then set it to
    ## over_rgba (if present)

    def _set_primary_scalars(self, arr, name):
        if self.scalar_data is not None \
               and self.scalar_data.size != arr.size:
            self.flush_arrays()
        rgba = quick_convert_rgba_to_vtk(arr)
        self.scalar_data = rgba
        self.scalar_name = name
        self.origin = self.blender.img_origin
        self.spacing = self.blender.img_spacing
        self.update()

    @t.on_trait_change('blender.main_rgba')
    def _set_main_array(self):
        # Set main_rgba into scalar_data.

        # changes
        # 1) from size 1 to size 1
        # 2) from size 1 to size 2
        # 3) from size 1 to 0 (with over_rgba)
        # 4) from size 1 to 0 (without over_rgba)
        # 5) from 0 to size 1


        print 'main rgba update', self.blender.main_rgba.size, self.blender.over_rgba.size
        # cases 1, 2, 5
        if self.blender.main_rgba.size:
            self._set_primary_scalars(self.blender.main_rgba, self.main_channel)

        # cases 3, 4 will be triggered if and when over_rgba changes

    @t.on_trait_change('blender.over_rgba')
    def _set_over_array(self):
        # Set over_rgba (and possibly blended_rgba) into appropriate arrays

        # cases
        # 1) main_rgba.size > 0
        # 2) main_rgba.size == 0
        print 'over rgba update', self.blender.over_rgba.size, self.blender.main_rgba.size
        if not self.blender.main_rgba.size and self.blender.over_rgba.size:
            self._set_primary_scalars(
                self.blender.over_rgba, self.over_channel
                )
            return
        if not self.blender.over_rgba.size:
            self.flush_arrays(names=[self.over_channel, self.blended_channel])
            return
        elif self.blender.over_rgba.size != self.blender.main_rgba.size:
            # this should always be prevented in the BlendedImages class
            raise RuntimeError('Color channel sizes do not match')

        # otherwise, append/set a new array with over_channel tag
        self.set_new_array(
            self.blender.over_rgba, self.over_channel, update=False
            )        
        # this obviously also changes the blended array
        self.set_new_array(
            self.blender.blended_rgba, self.blended_channel, update=True
            )

    def flush_arrays(self, names=[], update=True):
        pdata = self.image_data.point_data
        if not names:
            names = [pdata.get_array_name(n)
                     for n in xrange(pdata.number_of_arrays)]
        for n in names:
            pdata.remove_array(n)
        if update:
            self.image_data.update()
            self.image_data.update_traits()
            self.data_changed = True
            self.update()

    def set_new_array(self, arr, name, update=True):
        """

        Parameters
        ----------
        arr : 4-component ndarray with dtype = uint8
          The `arr` parameter must be shaped (npts x 4), or shaped
          (nz, ny, nx, 4) in C-order (like BlendedImage RGBA arrays when
          vtk_order is True)

        name : str
          name of the array
        
        """
        pdata = self.image_data.point_data
        if len(arr.shape) > 2:
            if len(arr.shape) > 3:
                flat_arr = arr.reshape(np.prod(arr.shape[:3]), 4)
            else:
                flat_arr = arr.ravel()
        else:
            flat_arr = arr
        chan = pdata.get_array(name)
        if chan:
            chan.from_array(flat_arr)
        else:
            n = pdata.add_array(flat_arr)
            pdata.get_array(n).name = name
        if update:
            self.image_data.update()
            # this one definitely needed
            self.pipeline_changed = True
            self.pipeline_changed = True

    
        
        
        
        
