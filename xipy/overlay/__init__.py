from PyQt4 import QtCore, QtGui
from enthought.traits import api as t_api
from nipy.core import api as ni_api
from xipy.slicing.image_slicers import VolumeSlicerInterface
from xipy.vis.qt4_widgets.auxiliary_window import TopLevelAuxiliaryWindow
import numpy as np
import matplotlib.cm as cm

class OverlayWindowInterface(TopLevelAuxiliaryWindow):
    """ This class defines the PyQt4 layer interface that an overlay
    system must conform to. From the point of view of any GUI element
    that wants to interface with an Overlay, this class provides two
    PyQt4 signals: loc_changed, and image_changed

    Spatial location updates cause emission of the loc_changed signal. To
    cause methods to be connected to this signal, include them in the
    loc_connections interable argument. Methods will have this signature
    pattern:
        meth(x, y, z)

    Overlay image updates are also signalled by the emission of the
    image_changed signal. To cause methods to be connected, include them in
    the image_connections iterable argument. Methods will have this signature
    pattern:
        meth(obj)

    * obj is a reference to the OverlayInterface type, giving access
      to a variety of overlay information (the OverlayInterface type is
      described elsewhere in this module)
    
    """
    # Whenever the voxel location focus changes, emit this signal
    # with the new location
    loc_changed = QtCore.pyqtSignal(float, float, float)
    # Whenever the overlay image itself changes, emit this signal
    # with a reference to the data managing object
    image_changed = QtCore.pyqtSignal(object)
    def __init__(self, loc_connections, image_connections, bbox,
                 parent=None, main_ref=None):
        """
        Creates a new OverlayWindow type
        """
        self.loc_connections = loc_connections
        self.image_connections = image_connections
        TopLevelAuxiliaryWindow.__init__(self, parent=parent, main_ref=main_ref)

    def _make_connections(self):
        for f in self.loc_connections:
            self.loc_changed.connect(f)
        for f in self.image_connections:
            self.image_changed.connect(f)
    def _break_connections(self):
        for f in self.loc_connections:
            self.loc_changed.disconnect(f)
        for f in self.image_connections:
            self.image_changed.disconnect(f)

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

# XYZ: SHOULD MAKE A TEST CLASS TO PROBAR ANY INSTANCE OF THIS INTERFACE
class OverlayInterface(t_api.HasTraits):
    """Different overlay managers will implement this interface
    """

    # an volume slicer for the voxel data
    overlay = t_api.Instance(VolumeSlicerInterface)

    # an event to say the overlay is updated
    overlay_updated = t_api.Event

    # the min/max value of the overlay to map to colors
    #norm = t_api.Tuple((0,1))
    norm = (0,1)

    # the alpha channel function for the colormap
    def alpha(self, scale=1.0, threshold=True):
        raise NotImplementedError

    # Color LUT options, and a reference to the matplotlib LUT
    cmap_option = t_api.Enum(*cm.cmap_d.keys())
    colormap = t_api.Property(depends_on='cmap_options')

    # Plotting interpolation options
    interpolation = t_api.Enum(['nearest', 'bilinear', 'sinc'])

    # thresholding information.. masks plotting of the overlay function
    # above/below a scalar value: for instance a plotter could set
    # the alpha channel to zero for scalar values less than 0 
    threshold = t_api.Tuple((0.0, 'inactive'))

    # fill value if using masked arrays
    fill_value = t_api.Float(0.0)

    # additional signal names, and lookup table
    _stats_maps = t_api.List
    stats_map = t_api.Enum(values='_stats_maps')
    stats_map_arrays = {}

    def _cmap_option_default(self):
        return 'jet'
    
    @t_api.cached_property
    def _get_colormap(self):
        return cm.cmap_d[self.cmap_option]

    def make_panel(self, parent=None):
        ui = self.edit_traits(parent=parent, kind='subpanel').control
        return ui

    def stats_overlay(self, stat_name):
        """Return a VolumeSlicer type for the requested stats map.
        It is assumed that the map has the same voxel to world mapping
        as the current overlay.

        Parameters
        ----------
        stat_name : str
            The name of the stats mapping to return as a VolumeSlicer type

        Returns
        -------
        a VolumeSlicerInterface subclass (of the same type as the
        current overlay)
        """
        if self.overlay is None:
            print 'Overlay not yet loaded'
            return None
        if stat_name not in self._stats_maps:
            print 'Stats array not loaded:', stat_name
            return None
        oclass = type(self.overlay)
        data = self.stats_map_arrays[stat_name]
        cmap = self.overlay.coordmap
        bbox = self.overlay.bbox # ???
        return oclass(ni_api.Image(data, cmap), bbox=bbox)


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
