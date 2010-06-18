from PyQt4 import QtCore, QtGui
import enthought.traits.api as t_api
import enthought.traits.ui.api as ui_api
from nipy.core import api as ni_api
from xipy.slicing.image_slicers import VolumeSlicerInterface
from xipy.vis.qt4_widgets.auxiliary_window import TopLevelAuxiliaryWindow
import xipy.volume_utils as vu
import xipy.vis.color_mapping as cm
import numpy as np

class OverlayWindowInterface(TopLevelAuxiliaryWindow):
    """ This class defines the PyQt4 layer interface that an overlay
    system must conform to.

    From the point of view of any GUI element that wants to interface
    with an Overlay, this class provides two PyQt4 signals:
    loc_changed, and image_changed

    Spatial location updates generated by interaction with the overlay
    cause emission of the loc_changed signal. To cause methods to be
    connected to this signal, include them in the loc_connections
    interable argument. Methods will have this signature pattern:
        meth(x, y, z)

    Overlay image updates are also signalled by the emission of the
    image_changed signal. To cause methods to be connected, include them in
    the image_connections iterable argument. Methods will have this signature
    pattern:
        meth(obj)

        * obj is a reference to the OverlayInterface type, giving access
          to a variety of overlay information (the OverlayInterface type is
          described elsewhere in this module)

    It may also be useful to monitor the voxel location from a different
    source, in which case the external_loc signal can be included in the
    constructor.
    
    """
    # Whenever the voxel location focus changes, emit this signal
    # with the new location
    loc_changed = QtCore.pyqtSignal(float, float, float)
    # Whenever the overlay image itself changes, emit this signal
    # with a reference to the data managing object
    image_changed = QtCore.pyqtSignal(object)
    # If the plot image mapping properties change, emit this signal
    # with a reference to the data managing object
    image_props_changed = QtCore.pyqtSignal(object)
    # A reference to the Traits-based data manager to be paired
    # with this window
    func_man = None
    # This will be the default registered name of the plugin tool
    tool_name = ''
    
    def __init__(self, loc_connections, image_connections,
                 props_connections, bbox,
                 external_loc=None, parent=None, main_ref=None):
        """
        Creates a new OverlayWindow type
        """
        self.loc_connections = loc_connections
        self.image_connections = image_connections
        self.props_connections = props_connections
        self.external_loc = external_loc
        TopLevelAuxiliaryWindow.__init__(self, window_name=self.tool_name,
                                         parent=parent, main_ref=main_ref)

    def _make_connections(self):
        for f in self.loc_connections:
            self.loc_changed.connect(f)
        for f in self.image_connections:
            self.image_changed.connect(f)
        for f in self.props_connections:
            self.image_props_changed.connect(f)
        
    def _break_connections(self):
        for f in self.loc_connections:
            self.loc_changed.disconnect(f)
        for f in self.image_connections:
            self.image_changed.disconnect(f)
        for f in self.props_connections:
            self.image_props_changed.disconnect(f)

    def activate(self):
        if self._activated:
            return
        self._make_connections()
        self._activated = True
    
    def deactivate(self, strip_overlay=False):
        if not self._activated:
            return
        if strip_overlay:
            self._strip_overlay()
        self._break_connections()
        self._activated = False

    def _strip_overlay(self):
        pass

    @classmethod
    def request_launch_callback(klass, launch_method):
        def launch_callback(*cb_args):
            launch_method(klass, *cb_args)
        return launch_callback


class ThresholdMap(t_api.HasTraits):
    map_voxels = t_api.Array
    map_scalars = t_api.Array

    # thresh limits are a pair of numbers that indicate the "good range"
    # of values in the scalar map. Depending on the mode, the mask will
    # work as follows:

    # * mask lower:
    #   Values lower than thresh_limits[0] will be masked out
    #
    # * mask higher:
    #   Values higher than thresh_limits[1] will be masked out
    #
    # * mask between:
    #   Values greater than thresh_limits[0] and less than thresh_limits[1]
    #   will be masked out
    #
    # * mask outside:
    #   Values less than thresh_limits[0] union values greater than
    #   thesh_limits[1] will be masked out
    thresh_limits = t_api.Tuple((0.0, 0.0))
    thresh_mode = t_api.Enum('mask lower', 'mask higher',
                             'mask between', 'mask outside')

    thresh_map_name = t_api.String

    # subclasses must fire this event whenever the mask must be recaculated
    map_changed = t_api.Event

    binary_mask = t_api.Property(depends_on='map_changed')

    @t_api.on_trait_change('map_scalars, thresh_limits, thresh_mode')
    def _dirty_mask(self):
        self.map_changed = True

    def _get_binary_mask(self):
        return self.create_binary_mask()

    def create_binary_mask(self, type='negative'):
        """Create a binary mask in the shape of map_scalars for the
        current threshold conditions.

        Parameters
        ----------
        type : str, optional
            By default, make a MaskedArray convention mask ('negative').
            Otherwise, set mask to True where values are unmasked ('positive')
        """
        if not self.thresh_map_name:
            return None
        mode = self.thresh_mode
        limits = self.thresh_limits
        map = self.map_scalars
        if mode=='mask lower':
            m = (map < limits[0]) if type=='negative' else (map >= limits[0])
        elif mode=='mask higher':
            m = (map > limits[1]) if type=='negative' else (map <= limits[1])
        elif mode=='mask between':
            m = ( (map > limits[0]) & (map < limits[1]) ) \
                if type=='negative' \
                else ( (map <= limits[0]) | (map >= limits[1]) )
        else: # mask outside
            m = ( (map < limits[0]) | (map > limits[1]) ) \
                if type=='negative' \
                else ( (map >= limits[0]) & (map <= limits[1]) )
        return m
    

# XYZ: SHOULD MAKE A TEST CLASS TO PROBAR ANY INSTANCE OF THIS INTERFACE
class OverlayInterface(t_api.HasTraits):
    """Different overlay managers will implement this interface
    """

    # Interface to keep track of the current world position in
    # an external plot, and a means to push a position back to listeners
    world_position = t_api.Array(shape=(3,))
    world_position_updated = t_api.Event
    def _world_position_default(self):
        return np.zeros(3)

    # VolumeSlicer for the voxel data
##     overlay = t_api.Instance(VolumeSlicerInterface)
    overlay = t_api.Instance(ni_api.Image)

    # Event to say the overlay is updated
    # XYZ: CAN'T TRAITS SIMPLY WATCH FOR "overlay" TO CHANGE?
    overlay_updated = t_api.Event

    # A text description of the overlay
    description = t_api.Any
    
    # Sclar-to-color mapping parameters:
    
    # 1) normalization parameters
    # the min/max value of the overlay to map to colors
    norm = t_api.Tuple( (0.0, 0.0) )

    # 2) RGBA lookup-table
    # Color LUT options, and a reference to the LUT object
    cmap_option = t_api.Enum(*sorted(cm.cmap_d.keys()))
    colormap = t_api.Property(depends_on='cmap_option')

    # 2a) a(x) function for scalar-to-alpha mapping
    def alpha(self, threshold=True):
        raise NotImplementedError

    # 3) interpolation
    interpolation = t_api.Enum(['nearest', 'bilinear', 'sinc'])

    # Signal for when the image should be redrawn
    image_props_updated = t_api.Event

    # Thresholding information
    threshold = t_api.Instance(ThresholdMap)

    # Fill value (if using masked arrays)
    fill_value = t_api.Float(0.0)

    # Additional stats map names, and lookup table
    _stats_maps = t_api.List
    stats_map = t_api.Enum(values='_stats_maps')

    # a simple UI group for the image property elements, may be used
    # or over-ridden in subclasses
    image_props_group = ui_api.Group(
        ui_api.Item('cmap_option', label='Colormaps'),
        ui_api.Item('interpolation', label='Interpolation Modes')
        )
    
    def _cmap_option_default(self):
        return 'jet'
    
    @t_api.cached_property
    def _get_colormap(self):
        return cm.cmap_d[self.cmap_option]

    def __init__(self, loc_signal=None, props_signal=None,
                 image_signal=None, **traits):
        """
        Parameters
        ----------
        loc_signal : QtCore.pyqtSignal
            optional PyQt4 callback signal to emit when peak finding
            (call pattern is loc_signal.emit(x,y,z))
        props_signal : QtCore.pyqtSignal
            optional PyQt4 callback signal to emit when image colormapping
            properties change
        image_signal : QtCore.pyqtSignal
            optional PyQt4 callback signal to emit when updating the image
            (call pattern is image_signal.emit(self))
        """
        self.image_signal = image_signal
        self.loc_signal = loc_signal
        self.props_signal = props_signal
        t_api.HasTraits.__init__(self, **traits)

    def make_panel(self, parent=None):
        ui = self.edit_traits(parent=parent, kind='subpanel').control
        return ui

    def map_stats_like_overlay(self, map_mask=False, mask_type='negative'):
        """Return a VolumeSlicer type for the current threshold scalar map.
        It is assumed that the map has the same voxel to world mapping
        as the current overlay.

        Returns
        -------
        a VolumeSlicerInterface subclass (of the same type as the
        current overlay)
        """
        if self.overlay is None:
            print 'Overlay not yet loaded'
            return None
        if self.threshold.thresh_map_name == '':
            print 'No active threshold'
            return None
        oclass = type(self.overlay)
        if map_mask:
            if mask_type=='negative':
                data = self.threshold.binary_mask.astype('d')
            else:
                data = (~self.threshold.binary_mask).astype('d')
        else:
            data = self.threshold.scalar_map
        cmap = self.overlay.coordmap
        bbox = self.overlay.bbox # ???
        grid_spacing = self.overlay.grid_spacing
        return oclass(ni_api.Image(data, cmap),
                      bbox=bbox, grid_spacing=grid_spacing)

def overlay_thresholding_function(threshold, positive=True):
    """ Take the OverlayInterface threshold parameters and create a
    function that maps from reals to {0,1}.

    Parameters
    ----------
    threshold : len-2 iterable
        the (threshold-value, comparison-type) pair thresholding parameters

    Returns
    -------
    func : the thresholding function.
        If positive is False, then the function will return a MaskedArray
        type field, where the points to mask evaluate to True
    """
    # from the interface class definition above, there will be 3 values
    # for the thresh type: inactive, less than, greater than
    t = threshold[0]
    if threshold[-1] == 'inactive':
        if positive:
            return lambda x: np.ones(x.shape, 'B')
        return lambda x: np.zeros(x.shape, 'B')
    elif threshold[-1] == 'less than':
        if positive:
            return lambda x: np.less(x,t)
        return lambda x: np.greater_equal(x,t)
    elif threshold[-1] == 'greater than':
        if positive:
            return lambda x: np.greater(x,t)
        return lambda x: np.less_equal(x,t)
    else:
        print 'unrecognized thresholding parameters:', threshold


def make_mpl_image_properties(func_man):
    """ Create a dictionary of matplotlib AxesImage color mapping
    properties from the corresponding properties in an OverlayInterface 
    """
    from matplotlib.colors import normalize
    props = dict()
    props['cmap'] = func_man.colormap
    props['interpolation'] = func_man.interpolation
    props['alpha'] = func_man.alpha()
    props['norm'] = normalize(*func_man.norm)
    return props
    
