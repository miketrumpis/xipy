"""This module has some local hackups of Mayavi objects"""
import numpy as np

from enthought.tvtk.api import tvtk
from enthought.tvtk.array_handler import get_vtk_array_type
import enthought.mayavi.sources.api as src_api
from enthought.traits.api import Trait, Instance, CInt, Range, Enum
from enthought.mayavi.modules.image_plane_widget import ImagePlaneWidget
from enthought.mayavi.filters.set_active_attribute import SetActiveAttribute
from enthought.mayavi.tools.modules import ImagePlaneWidgetFactory
from enthought.mayavi.tools.filters import SetActiveAttributeFactory
from enthought.mayavi.tools.pipe_base import make_function

import enthought.traits.api as t
from enthought.mayavi.sources.vtk_data_source import VTKDataSource, \
     has_attributes, get_all_attributes
from enthought.tvtk.api import tvtk

# -- XIPY imports
from xipy.colors.rgba_blending import BlendedImages, quick_convert_rgba_to_vtk

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
    """Validates a scalar array passed to the object."""
    if value is None:
        return None
    arr = np.asarray(value)
    # make two branches,
    #  * for vtkUnsignedChar (numpy's uint8, or 'B') -- this will
    #    become a 4-component color mapped array
    #  * for scalar valued 2- or 3-D arrays
    if arr.dtype.char == 'B':
        assert arr.ndim == 4, \
               "Color mapped RGBA arrays must be 4-dimensional"
        assert arr.shape[-1] == 4, \
               "Color component dimension for scalar array must be length-4"
        xyz_dims = arr.shape[:-1]
    else:
        assert arr.ndim in [2,3], "Scalar array must be 2 or 3 dimensional"
        xyz_dims = arr.shape
    vd = obj.vector_data
    if vd is not None:
        assert vd.shape[:-1] == xyz_dims, \
               "Scalar array must match already set vector data.\n"\
               "vector_data.shape = %s, given array shape = %s"%(vd.shape,
                                                                 xyz_dims)
    return arr

_check_scalar_array.info = 'a 2-, 3-, or 4D numpy array'

class ArraySourceRGBA(src_api.ArraySource):
    """This is a version of ArraySource that allows for assignment
    of RGBA component data -- IE,
    data.shape must be (nx, ny, [nz, 4])
    """
    scalar_data = Trait(None, _check_scalar_array, rich_compare=True)

    def _scalar_data_changed(self, data):
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

        flat_shape = ( np.prod(dims), )
        if is_rgba_bytes:
            flat_shape += (4,)
        if self.transpose_input_array:
            if is_rgba_bytes:
                # keep the color components in the last dimension
                d = data.transpose(2,1,0,3).copy()
                d.shape = flat_shape
                img_data.point_data.scalars = d
            else:
                img_data.point_data.scalars = np.ravel(np.transpose(data))
            img_data.dimensions = tuple(dims)
            img_data.extent = 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1
            img_data.update_extent = 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1
        else:
            img_data.point_data.scalars = data.reshape(flat_shape)
            img_data.dimensions = tuple(dims[::-1])
            img_data.extent = 0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1
            img_data.update_extent = 0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1

        img_data.number_of_scalar_components = 4 if is_rgba_bytes else 1
        img_data.point_data.scalars.name = self.scalar_name
        # This is very important and if not done can lead to a segfault!
        typecode = data.dtype
        img_data.scalar_type = get_vtk_array_type(typecode)
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

    @t.on_trait_change('use_lookup_table')
    def setup_lut(self): 
        # Set the LUT for the IPW.
        if self.use_lookup_table:
            if self.module_manager is not None:
                self.ipw.lookup_table = \
                                self.module_manager.scalar_lut_manager.lut
        else:
            self.module_manager._teardown_event_handlers()
            self.ipw.color_map.lookup_table = None
        self.render()

class ImagePlaneWidgetFactory_RGBA(ImagePlaneWidgetFactory):
    """ Applies the ImagePlaneWidget mayavi module to the given data
        source (Mayavi source, or VTK dataset). 
    """
    _target = Instance(ImagePlaneWidget_RGBA, ())

image_plane_widget_rgba = make_function(ImagePlaneWidgetFactory_RGBA)

# -- An ArraySource with "channels" ------------------------------------------

def disable_render(method):
    def wrapped_method(obj, *args, **kwargs):
        try:
            render_state = obj.scene.disable_render
            obj.scene.disable_render = True
        except AttributeError:
            pass
        res = method(obj, *args, **kwargs)
        try:
            obj.scene.disable_render = render_state
        except:
            pass
        return res
    for attr in ['func_doc', 'func_name']:
        setattr(wrapped_method, attr, getattr(method, attr))
    return wrapped_method

class MasterSource(VTKDataSource):

    """
    This class monitors a BlendedImages object and sets up image
    channels for the main image, over image, and blended image.

    It may have additional channels

    This class subclasses VTKDataSource as a handy base class for a
    "has-a" tvtk dataset design--but there are some drawbacks as well.
    """

    data = t.Instance(tvtk.ImageData, args=(), allow_none=False)

    # XXX: This source should have a signal saying when a channel may
    # disappear!!! EG, if blender.over gets set to None, and some module
    # is visualizing "over_colors", then there will be a big fat crash!!
    
    blender = t.Instance(BlendedImages)
    over_channel = 'over_colors'
    main_channel = 'main_colors'
    blended_channel = 'blended_colors'

    colors_changed = t.Event
    
    rgba_channels = t.Property
    all_channels = t.Property

    def __init__(self, *args, **kwargs):
        super(MasterSource, self).__init__(*args, **kwargs)
        self.data = tvtk.ImageData()

    @t.on_trait_change('blender')
    def _check_vtk_order(self):
        if not self.blender.vtk_order:
            raise ValueError('BlendedImages instance must be in VTK order')

    def _get_all_channels(self):
        pdata = self.data.point_data
        names = [pdata.get_array_name(n)
                 for n in xrange(pdata.number_of_arrays)]
        return names

    def _get_rgba_channels(self):
        primary_channels = (self.over_channel,
                            self.main_channel,
                            self.blended_channel)
        names = self.all_channels
        return [n for n in names if n in primary_channels]

    @t.on_trait_change('blender.main_rgba')
    def _set_main_array(self):
        # Set main_rgba into scalar_data.

        # changes
        # 1) from size 1 to size 1
        # 2) from size 1 to size 2
        # 3) from size 1 to 0 (with over_rgba)
        # 4) from size 1 to 0 (without over_rgba)
        # 5) from 0 to size 1


        # cases 1, 2, 5
        if self.blender.main_rgba.size:
            self._change_primary_scalars(self.blender.main_rgba,
                                         self.main_channel)

        elif not self.blender.over_rgba.size:
            #self.flush_arrays()
            self.safe_remove_arrays()
        # cases 3, 4 will be triggered if and when over_rgba changes

    @t.on_trait_change('blender.over_rgba')
    def _set_over_array(self):
        # Set over_rgba (and possibly blended_rgba) into appropriate arrays

        # cases
        # 1) main_rgba.size > 0
        # 2) main_rgba.size == 0
        if not self.blender.main_rgba.size and self.blender.over_rgba.size:
            self._change_primary_scalars(
                self.blender.over_rgba, self.over_channel
                )
            return
        if not self.blender.over_rgba.size:
            #self.flush_arrays(names=[self.over_channel, self.blended_channel])
            self.safe_remove_arrays(names=[self.over_channel,
                                           self.blended_channel])
            return
        elif self.blender.over_rgba.size != self.blender.main_rgba.size:
            # this should always be prevented in the BlendedImages class
            raise RuntimeError('Color channel sizes do not match')

        # otherwise, append/set a new array with over_channel tag
        updating = True #self.over_channel not in self.all_channels

        print 'should update over rgba, blended rgba'
        self.set_new_array(
            self.blender.over_rgba, self.over_channel, update=False
            )        
        # this obviously also changes the blended array
        self.set_new_array(
            self.blender.blended_rgba, self.blended_channel
            )

    def _push_changes(self):
        # this should be called when..
        # * arrays are added/removed
        self._update_data()
        self.pipeline_changed = True
        self.colors_changed = True
##         self.pipeline_changed = True
        
    def _check_aa(self):
        aa = self._assign_attribute
        if has_attributes(self.data) and not aa.input:
            aa = self._assign_attribute
            aa.input = self.data
            self._update_data()
            self.outputs = [aa.output]
        else:
            self.outputs = [self.data]

    ## The convention will be to have main_rgba be the primary array
    ## in scalar_data. If main_rgba isn't present, then set it to
    ## over_rgba (if present)

    def _change_primary_scalars(self, arr, name):
        """

        Parameters
        ----------

        arr: ndarray, shape (Nx, Ny, Nz, 4)
           If this is going in as primary scalars, it is definitely an RGBA
           vector array provided by a BlendedImages. Therefore is needs to
           be reshaped to C-order (Nz, Ny, Nx, 4)

        name: str
           array label
        """
        pd = self.data.point_data
        if pd.scalars is not None \
               and pd.scalars.size != arr.size:
            #self.flush_arrays(update=False)
            self.safe_remove_arrays()
        rgba = quick_convert_rgba_to_vtk(arr)
        xyz_shape = rgba.shape[:3]
        flat_shape = (np.prod(xyz_shape), 4)
        dataset = self.data


        # set the ImageData metadata
        dataset.origin = self.blender.img_origin
        dataset.spacing = self.blender.img_spacing
        dataset.dimensions = xyz_shape[::-1]
        dataset.extent = 0, xyz_shape[2]-1, 0, xyz_shape[1]-1, 0, xyz_shape[0]-1
        dataset.update_extent = dataset.extent

        dataset.number_of_scalar_components = 4
        dataset.scalar_type = get_vtk_array_type(arr.dtype)

        # set the scalars and name
        self.set_new_array(rgba, name, update=False)
##         pd.scalars = rgba.reshape(flat_shape)
##         pd.scalars.name = name

        dataset.update()
        dataset.update_traits()
        self._check_aa()
        self._update_data()
        self._push_changes()
        self.point_scalars_name = name

    @disable_render
    def safe_remove_arrays(self, names=[]):
        if not names:
            names = self.all_channels
        for name in names:
            # 1st, determine if the name is being used by any child
            l = [self]
            while l:
                # dequeue the 1st node
                node = l.pop(0)
                if hasattr(node, 'children') and node.children:
                    # enqueue the children nodes
                    l += node.children
                # examine this node
                used_name = getattr(node, 'point_scalars_name', None)
                if used_name == name:
                    node.point_scalars_name = ''
##                     node.stop()
            # now remove the array safely
            self.data.point_data.remove_array(name)

        # XXX: is this right?
        self._push_changes()

    @disable_render
    def set_new_array(self, arr, name, update=True):
        """
        Sets up a new point data channel in this object's ImageData

        Parameters
        ----------
        arr : 4-component ndarray with dtype = uint8
          The `arr` parameter must be shaped (npts x 4), or shaped
          (nz, ny, nx, 4) in C-order (like BlendedImage RGBA arrays when
          vtk_order is True)

        name : str
          name of the array
        
        """
        pdata = self.data.point_data
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
            self._push_changes()

    
    # -------- BUG FIX??
    def _update_data(self):
        if self.data is None:
            return
        pnt_attr, cell_attr = get_all_attributes(self.data)
        
        def _setup_data_traits(obj, attributes, d_type):
            """Given the object, the dict of the attributes from the
            `get_all_attributes` function and the data type
            (point/cell) data this will setup the object and the data.
            """
            attrs = ['scalars', 'vectors', 'tensors']
            aa = obj._assign_attribute
            data = getattr(obj.data, '%s_data'%d_type)
            for attr in attrs:
                values = attributes[attr]
                values.append('')
                setattr(obj, '_%s_%s_list'%(d_type, attr), values)
                if len(values) > 1:
                    default = getattr(obj, '%s_%s_name'%(d_type, attr))
                    if obj._first and len(default) == 0:
                        default = values[0]
                    getattr(data, 'set_active_%s'%attr)(default)
                    aa.assign(default, attr.upper(),
                              d_type.upper() +'_DATA')
                    aa.update()
                    kw = {'%s_%s_name'%(d_type, attr): default,
                          'trait_change_notify': False}
                    obj.set(**kw)

        _setup_data_traits(self, pnt_attr, 'point')
        _setup_data_traits(self, cell_attr, 'cell')


        pd = self.data.point_data
        scalars = pd.scalars
        if self.data.is_a('vtkImageData') and scalars is not None:
            # For some reason getting the range of the scalars flushes
            # the data through to prevent some really strange errors
            # when using an ImagePlaneWidget.
            r = scalars.range
            self._assign_attribute.output.scalar_type = scalars.data_type
            self.data.scalar_type = scalars.data_type
            self._assign_attribute.output.update_traits()
##             self.data.update_traits()

        if self._first:
            self._first = False
        # Propagate the data changed event.
        self.data_changed = True    

# -------- FIX TO SET_ACTIVE_ATTRIBUTE??

class SetImageActiveAttribute(SetActiveAttribute):

    def _setup_output(self):
        idata = self.inputs[0].outputs[0]
        odata = self.outputs[0]
        if idata.is_a('vtkImageData') and odata.is_a('vtkImageData'):
            for dt in ('point', 'cell'):
                scalars_name = getattr(self, dt+'_scalars_name')
                input_data = getattr(idata, dt+'_data', None)
                if input_data is not None:
                    input_scalars = input_data.scalars
                    if input_scalars and input_scalars.name == scalars_name:
                        print 'setting dtype'
                        odata.scalar_type = input_scalars.data_type

    def update_data(self):
        print 'setting up output from update_data'
        self._setup_output()
        SetActiveAttribute.update_data(self)
    
    def update_pipeline(self):
##         super(SetImageActiveAttribute, self).update_pipeline()
        SetActiveAttribute.update_pipeline(self)
        print 'setting up output from update_pipeline'
        self._setup_output()
        
class SetImageActiveAttributeFactory(SetActiveAttributeFactory):
    """ Applies the SetActiveAttribute Filter mayavi filter to the given 
    VTK object.
    """
    _target = Instance(SetImageActiveAttribute, ())


set_image_active_attribute = make_function(SetImageActiveAttributeFactory)
