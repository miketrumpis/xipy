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
        print 'data changed'
        img_data = self.image_data
        if data is None:
            img_data.point_data.scalars = None
            self.data_changed = True
            return
        dims = list(data.shape[:-1])
        if len(dims)==2:
            dims.append(1)
      
        img_data.origin = tuple(self.origin)
        img_data.dimensions = tuple(dims)
        img_data.extent = 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1
        img_data.update_extent = 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1
        if self.transpose_input_array:
            d = data.transpose(2,1,0,3).copy()
            d.shape = ( np.prod(d.shape[:3]), 4 )
            img_data.point_data.scalars = d
        else:
            d = data.reshape( np.prod(data.shape[:3]), 4 )
            img_data.point_data.scalars = d
        img_data.number_of_scalar_components = 4
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


if __name__ == '__main__':
    from xipy.vis import rgba_blending
##     from matplotlib import cm
    import xipy.vis.color_mapping as cm
    from enthought.mayavi import mlab

    #### Some synthetic data
    fx = 30.0; fy = 74.0; fz = 20.0
    sw_3d = np.sin(2*np.pi*( np.arange(128)[:,None,None]/fx + \
                             np.arange(128)[None,:,None]/fy + \
                             np.arange(128)[None,None,:]/fz ))
    win = np.zeros(128)
    win[32:96] = np.hanning(64)
    win_3d = np.power(win[:,None,None] * win[None,:,None] * win[None,None,:], 1/3.)
    sw_3d *= win_3d
    main_dr = np.array([1.0]*3)
    main_r0 = np.array([-64.]*3)
    
    rn_3d = np.random.randn(37,49,80)
    over_dr = (np.array(sw_3d.shape,'i')/np.array(rn_3d.shape,'i')).astype('d')
    over_r0 = -(over_dr*rn_3d.shape)/2

    #### Colormap the scalar data
    main_bytes1 = rgba_blending.normalize_and_map(sw_3d, cm.gray)
    main_bytes2 = main_bytes1.copy()
    
    over_bytes1 = rgba_blending.normalize_and_map(rn_3d, cm.hot, alpha=.25)
    # also with an alpha function, rather than a constant alpha--
    # emphasizes larger + and - numbers
    mn = rn_3d.min(); mx = rn_3d.max()
    lx = np.linspace(mn, mx, 256)
    lx *= 2*np.pi/max(abs(mn), abs(mx))
    afunc = np.abs(np.arctan(lx)) * (255*2/np.pi)
    over_bytes2 = rgba_blending.normalize_and_map(rn_3d, cm.jet, alpha=afunc)
    
    rgba_blending.resample_and_blend(main_bytes1, main_dr, main_r0,
                                     over_bytes1, over_dr, over_r0)
    rgba_blending.resample_and_blend(main_bytes2, main_dr, main_r0,
                                     over_bytes2, over_dr, over_r0)

    #### Make the Mayavi sources
##     src1 = ArraySourceRGBA(transpose_input_array=False)
    src1 = src_api.ArraySource(transpose_input_array=False)
    src1.scalar_data = main_bytes1[60]
##     src2 = ArraySourceRGBA(transpose_input_array=False)
    src2 = src_api.ArraySource(transpose_input_array=False)
    src2.scalar_data = main_bytes2[60]

    src1 = mlab.pipeline.add_dataset(src1)
    src2 = mlab.pipeline.add_dataset(src2)

##     ipw1 = image_plane_widget_rgba(src1)
    ipw1 = mlab.pipeline.image_plane_widget(src1); ipw1.use_lookup_table = False
    ipw1.ipw.plane_orientation = 'z_axes'
    ipw1.ipw.slice_index = 30
##     ipw2 = image_plane_widget_rgba(src2)
    ipw2 = mlab.pipeline.image_plane_widget(src2); ipw2.use_lookup_table = False
    ipw2.ipw.plane_orientation = 'x_axes'
    ipw2.ipw.slice_index = 20
    mlab.show()
    
