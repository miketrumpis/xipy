import os
os.environ['ETS_TOOLKIT'] = 'qt4'

import numpy as np
from PyQt4 import QtGui, QtCore

from enthought.traits.api \
    import HasTraits, HasPrivateTraits, Instance, Enum, Dict, Constant, Str, \
    List, on_trait_change, Float, File, Array, Button, Range, Property, \
    cached_property, Event, Bool, Color, Int, String
    
from enthought.traits.ui.api \
    import Item, Group, View, VGroup, HGroup, HSplit, \
    EnumEditor, CheckListEditor, ListEditor, message, ButtonEditor, RangeEditor

from enthought.traits.ui.file_dialog import open_file

from xipy.slicing import load_sampled_slicer, load_resampled_slicer
from xipy.slicing.image_slicers import SampledVolumeSlicer, \
     ResampledVolumeSlicer
from xipy.vis.qt4_widgets import browse_files
from xipy.vis.qt4_widgets.colorbar_panel import ColorbarPanel
from xipy.overlay import OverlayInterface, OverlayWindowInterface
from xipy.volume_utils import signal_array_to_masked_vol

from nipy.core import api as ni_api
from nipy.core.reference.coordinate_map import compose

import matplotlib as mpl

class ImageOverlayWindow( OverlayWindowInterface ):
    """ This QT4 window is a essentially a frame to hold a colorbar
    representation of the image overlay's colormap, and the image
    overlay manager's UI control panel. It also creates some PyQt4
    signals which can be monitored by other UI elements.
    """

    tool_name = 'Image Overlay Controls'

    def __init__(self, loc_connections, image_connections,
                 image_props_connections, bbox,
                 overlay=None, external_loc=None,
                 parent=None, main_ref=None):
        OverlayWindowInterface.__init__(self,
                                        loc_connections,
                                        image_connections,
                                        image_props_connections,
                                        bbox, # <-- TRY TO DECOUPLE THIS
                                        external_loc=external_loc,
                                        parent=parent,
                                        main_ref=main_ref)
        vbox = QtGui.QVBoxLayout(self)
        self.cbar = ColorbarPanel(parent=self, figsize=(6,2))

        vbox.addWidget(self.cbar)
        
        self.func_man = ImageOverlayManager(
            bbox, colorbar=self.cbar,
            loc_signal=self.loc_changed,
            image_signal=self.image_changed,
            props_signal=self.image_props_changed,
            overlay=overlay
            )
        
        vbox.addWidget(self.func_man.make_panel(parent=self))
        QtGui.QWidget.setSizePolicy(self,
                                    QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Expanding)
        self.updateGeometry()
        r = self.geometry()
        self.cbar.fig.set_size_inches(r.width()/100., r.height()/200.)
        self.cbar.fig.canvas.draw()
        
    def _strip_overlay(self):
        self.cbar.initialize()

class ImageOverlayManager( OverlayInterface ):
    """ A data management class to interact with an image
    """

    _new_overlay = Event
    lbutton = Button('Load Overlay Image')
    # Localizer functions
    _xform_by_name = {
        'max' : lambda x: np.argmax(x),
        'min' : lambda x: np.argmin(x),
        'absmax' : lambda x: np.argmax(np.abs(x))
        }
    ana_xform = Enum(['max', 'min', 'absmax'])
    loc_button = Button('Find Extremum')
    # this number controls searching for the Nth highest/lowest value
    _numfeatures = Int(150)
    _one = Int(1)
    order = Range('_one', '_numfeatures')
    
    tval = Range(low='min_t', high='max_t',
                 editor=RangeEditor(low_name='min_t', high_name='max_t',
                                    low_label='min_t_label',
                                    high_label='max_t_label',
                                    format='%1.2f'))
    comp = Enum('greater than', 'less than')

    mask = Property(Array, depends_on='tval, comp, _new_overlay')
    # XYZ: RETHINK WHEN ordered_idx SHOULD BE RECALCULATED.. MAYBE ONLY
    # WHEN THE THE MASK IS "DIRTY" (IE RECENTLY APPLIED, BUT NOT USED)
    work_arr = Property(Array, depends_on='tval, _new_overlay')
    ordered_idx = Property(Array, depends_on='_new_overlay, tval, ana_xform')

    max_t = Property(depends_on='_new_overlay')
    max_t_label = Property(depends_on='_new_overlay')
    min_t = Property(depends_on='_new_overlay')
    min_t_label = Property(depends_on='_new_overlay')
    mask_button = Button('Apply Mask')
    clear_button = Button('Clear Mask')

    grid_size = Range(1,150)

    peak_color = Color

    # these are just to satisfy the informal "overlay manager" interface
    fill_value = np.nan
    description = String('An Overlay!')

    _base_alpha = np.ones((256,), dtype='B')

    def alpha(self, scale=1.0, threshold=True):
        # scale may go between 0 and 4.. just map this from (0,1)
        a = (scale/4.0)*self._base_alpha
        if threshold and self.threshold[1] != 'inactive':
            tval, comp = self.threshold
            mn, mx = self.norm
            lut_map = int(255 * (tval - mn)/(mx-mn))
            if comp == 'greater than':
                a[:lut_map] = 0
            else:
                a[lut_map+1:] = 0
        return a

    def __init__(self, bbox, colorbar=None,
                 loc_signal=None, image_signal=None, props_signal=None,
                 overlay=None, **traits):
        """
        Parameters
        ----------
        bbox : iterable
            the {x,y,z} limits of the volume in which to plot an overlay
        colorbar : ColorbarPanel object (optional)
            a Qt4 widget with a color table mapping the overlay values
        loc_signal : QtCore.pyqtSignal (optional)
            optional PyQt4 callback signal to emit when peak finding
            (call pattern is loc_signal.emit(x,y,z))
        image_signal : QtCore.pyqtSignal (optional)
            optional PyQt4 callback signal to emit when updating the image
            (call pattern is image_signal.emit(self))
        props_signal : QtCore.pyqtSignal (optional)
            optional PyQt4 callback signal to emit when only updating
            image mapping properties
        overlay : str, NIPY Image, VolumeSlicer type (optional)
            some version of the data to be overlaid
        """
        HasTraits.__init__(self, **traits)
        # this is a necessary argument when creating any new overlays
        self.bbox = bbox
        self._new_overlay = False
        self.cbar = colorbar
        self._loc_signal = loc_signal
        self._image_signal = image_signal
        self._props_signal = props_signal
        if overlay:
            self.update_overlay(overlay)
        else:
            self.overlay is None

    def update_overlay(self, overlay, silently=False):
##         self.overlay = load_sampled_slicer(overlay, self.bbox)
        self.overlay = load_resampled_slicer(overlay, self.bbox)
        print 'world bbox:', self.bbox
        print 'overlay bbox:', self.overlay.bbox
        if self.overlay.raw_mask is not None:
            self.orig_mask = np.logical_not(self.overlay.raw_mask)
        else:
            self.orig_mask = np.zeros(self.overlay.raw_image.shape, np.bool)
        # this will be the array used for peak finding and threshold
        # setting. It is the "original" data, from the overlay slicer
        # object. Furthermore, all peak finding will be done on the
        # array's index coordinates, and not on the interpolated
        # data maps
        self.m_arr = np.ma.masked_array(np.asarray(self.overlay.raw_image),
                                        self.orig_mask,
                                        copy=False)
        # toggling False to True should reset properties???
        self._new_overlay = True 
        print 'normalizing from', self.min_t, 'to', self.max_t
        self.norm = (self.min_t, self.max_t)
        print self.norm
        if self.cbar:
            self.cbar.change_norm(mpl.colors.normalize(*self.norm))
        if not silently:
            self.send_image_signal()

    def send_image_signal(self):
        self.overlay_updated = True
        if self._image_signal:
            self._image_signal.emit(self)

    def send_location_signal(self, loc):
        if self._loc_signal:
            self._loc_signal.emit(*loc)

    @on_trait_change('norm, cmap_option, interpolation')
    def send_props_signal(self):
        if self._props_signal:
            self._props_signal.emit(self)
    
    ### CALLBACKS
    def _lbutton_fired(self):
        f = browse_files(None, dialog='Select File',
                         wildcard='*.nii *.nii.gz *.hdr *.img')
        if f:
            self.update_overlay(f)

    def _loc_button_fired(self):
        # XYZ: DON'T THINK THIS WILL EXECUTE IN A NEW THREAD: FIX
        self.find_peak()

    def _mask_button_fired(self):
        self.threshold = (self.tval, self.comp)
        self.send_props_signal()
##         self.create_mask()

    def _clear_button_fired(self):
        self.threshold = (self.tval, 'inactive')
        self.send_props_signal()

    @on_trait_change('order') #, dispatch='new')
    def find_peak(self):
        if self.overlay is None:
            return
        if self.ana_xform in ('absmax', 'max'):
            pk_flat_idx = self.ordered_idx[-self.order]
        else:
            pk_flat_idx = self.ordered_idx[self.order-1]

        vol_idx = np.array(np.lib.index_tricks.unravel_index(pk_flat_idx,
                                                             self.m_arr.shape))
        
        xyz_a = self.overlay.coordmap(vol_idx)[0]
        xyz_b = self.overlay.coordmap(vol_idx+1)[0]
        pk_val_arr = self.m_arr[tuple(vol_idx)]
        print xyz_a, xyz_b
        print 'vox coords:', vol_idx, 'should cut to coords', (xyz_a+xyz_b)/2
        print 'peak value:', pk_val_arr
        if self.cbar:
##             rgb = self.cbar.cb.(pk_val)
##             rgb_255 = map(lambda x: int(round(255*x)), rgb[:3])
##             print rgb_255
            hx = self.cbar(pk_val_arr)
            self.peak_color = hx
        self.send_location_signal((xyz_a + xyz_b)/2)

    @on_trait_change('tval')
    def move_cbar_indicator(self):
        if not self.cbar:
            return
        self.cbar.change_threshold(self.tval)

    @on_trait_change('grid_size')
    def new_grid_size(self):
        # could also use the overlay.update_grid_spacing function
        img = self.overlay.raw_image
##         mask = self.mask
##         new_slicer = SampledVolumeSlicer(img, bbox=self.bbox,
##                                          grid_spacing=[float(self.grid_size)]*3)
        new_slicer = ResampledVolumeSlicer(
            img, bbox=self.bbox, grid_spacing=[float(self.grid_size)]*3
            )
        self.update_overlay(new_slicer)

    ### PROPERTY FUNCTIONS
    @cached_property
    def _get_max_t(self):
        print 'recomputing max_t'
        if self.overlay is None:
            return 0.0
        mx = self.m_arr.max()
        return int(1000*mx)/1000.0
    @cached_property
    def _get_max_t_label(self):
        return '%1.2f'%self.max_t
    @cached_property
    def _get_min_t(self):
        print 'recomputing min_t'
        if self.overlay is None:
            return 0.0
        mn = self.m_arr.min()
        return int(1000*mn)/1000.0
    @cached_property
    def _get_min_t_label(self):
        return '%1.2f'%self.min_t
    @cached_property
    def _get_mask(self):
        """ Create a negative mask of the overlay map, where points
        masked are marked as True
        """
        if self.overlay is None:
            return None
        om = self.orig_mask # neg mask
        if self.comp=='greater than':
            # mask True for all vals less than or equal to threshold
            nm = self.m_arr.data <= self.tval # neg mask
        else:
            # mask True all vals greater than or equal to threshold
            nm = self.m_arr.data >= self.tval
        m = nm | om
        return m

    @cached_property
    def _get_work_arr(self):
        if self.overlay is None:
            return None
        return np.ma.masked_array(self.m_arr.data, mask=self.mask, copy=False)
    @cached_property
    def _get_ordered_idx(self):
        """ Create a list of sorted map indices
        """
        if self.overlay is None:
            return None
        m_arr = np.abs(self.work_arr) if self.ana_xform=='absmax' \
                else self.work_arr
        sidx = m_arr.flatten().argsort()
        last_good = m_arr.mask.flat[sidx].nonzero()[0][0]
        self._numfeatures = last_good
        return sidx[:last_good]

    view = View(
        HGroup(
            VGroup(
                HGroup(
                    Item('lbutton', show_label=False),
                    Item('grid_size', label='Grid Size')
                    ),
                Item('_'),
                HGroup(
                    Item('cmap_option', label='Colormap'),
                    Item('interpolation', label='Interpolation'),
                    ),
                Item('_'),
                Item('comp', label='Threshold Comparison'),
                Item('tval', style='custom', label='Overlay Threshold'),
                HGroup(
                    Item('mask_button', show_label=False),
                    Item('clear_button', show_label=False)
                    )
                ),
            VGroup(Item('ana_xform', label='Feature Transform', width=10),
                   HGroup(Item('loc_button', show_label=False),
                          Item('order', style='simple')
                          ),
                   Item('peak_color', style='readonly')
                   )
            ),
        resizable=True,
        title='Image Overlay Controls'
        )

## register_overlay_plugin('Image Overlay', ImageOverlayWindow)

if __name__=='__main__':
    bbox = [(-100,100)]*3
    overmanager = ImageOverlayManager(bbox)
    overmanager.configure_traits()
