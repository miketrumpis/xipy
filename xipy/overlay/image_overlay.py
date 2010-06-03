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
     ResampledVolumeSlicer, ResampledIndexVolumeSlicer
from xipy.vis.qt4_widgets import browse_files
from xipy.vis.qt4_widgets.colorbar_panel import ColorbarPanel
from xipy.overlay import OverlayInterface, OverlayWindowInterface, ThresholdMap
from xipy.volume_utils import signal_array_to_masked_vol
from xipy.io import load_image

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
                 functional_manager=None,
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
        if functional_manager is None or \
               type(functional_manager) is not ImageOverlayManager:
            self.func_man = ImageOverlayManager(
                bbox, colorbar=self.cbar,
                loc_signal=self.loc_changed,
                image_signal=self.image_changed,
                props_signal=self.image_props_changed,
                overlay=overlay
                )
        else:
            self.func_man = functional_manager
            self.func_man.loc_signal=self.loc_changed
            self.func_man.image_signal=self.image_changed
            self.func_man.props_signal=self.image_props_changed
            self.func_man.connect_colorbar(self.cbar)
            # breaking independence
            if self.func_man.overlay is not None:
                print 'requesting new image signal'
                self.func_man.send_image_signal()
        
        vbox.addWidget(self.func_man.make_panel(parent=self))
        QtGui.QWidget.setSizePolicy(self,
                                    QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Expanding)
        self.updateGeometry()
        r = self.geometry()
        self.cbar.figure.set_size_inches(r.width()/100., r.height()/200.)
        self.cbar.draw()
        
    def _strip_overlay(self):
        self.cbar.initialize()

class ImageOverlayManager( OverlayInterface ):
    """ A data management class to interact with an image
    """
    lbutton = Button('Load Overlay Image')

    raw_image = Instance(ni_api.Image)

    # Peak finding
    ana_xform = Enum(['max', 'min', 'absmax'])
    loc_button = Button('Find Extremum')
    # this number controls searching for the Nth highest/lowest value
    _numfeatures = Int(150)
    _one = Int(1)
    order = Range('_one', '_numfeatures')

    _recompute_props = Event

    tval = Range(low='_min_t', high='_max_t',
                 editor=RangeEditor(low_name='_min_t', high_name='_max_t',
                                    #low_label='min_t_label',
                                    #high_label='max_t_label',
                                    format='%1.2f'))
    comp = Enum('greater than', 'less than')

    mask = Property(Array, depends_on='tval, comp, _recompute_props')
    # XYZ: RETHINK WHEN ordered_idx SHOULD BE RECALCULATED.. MAYBE ONLY
    # WHEN THE THE MASK IS "DIRTY" (IE RECENTLY APPLIED, BUT NOT USED)
    work_arr = Property(Array, depends_on='tval, _recompute_props')
    ordered_idx = Property(Array,
                           depends_on='_recompute_props, tval, ana_xform')

    _min_t = Float
    _max_t = Float

    mask_button = Button('Apply Mask')
    clear_button = Button('Clear Mask')

    grid_size = Range(1,150)

    peak_color = Color

    description = Property(depends_on='overlay_updated, tval, mask_button, clear_button')

    # these are just to satisfy the informal "overlay manager" interface
    fill_value = np.nan

    _base_alpha = 2*np.ones((256,), dtype='B')

    def alpha(self, scale=1.0, threshold=True):
        # scale may go between 0 and 4.. just map this from (0,1)
        a = (scale/4.0)*self._base_alpha
##         if threshold and self.threshold[1] != 'inactive':
##             tval, comp = self.threshold
##             mn, mx = self.norm
##             lut_map = int(255 * (tval - mn)/(mx-mn))
##             if comp == 'greater than':
##                 a[:lut_map] = 0
##             else:
##                 a[lut_map+1:] = 0
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
        OverlayInterface.__init__(self,
                                  loc_signal=loc_signal,
                                  props_signal=props_signal,
                                  image_signal=image_signal,
                                  **traits)
        self.bbox = bbox
        self.threshold = ThresholdMap()
        self.connect_colorbar(colorbar)
        if overlay:
            self.update_overlay(overlay)
        else:
            self.overlay is None

    def update_overlay(self, overlay, silently=False):
        if type(overlay)==ni_api.Image:
            raw_image = overlay
        else:
            raw_image = load_image(overlay)
                
        self.orig_mask = np.ma.getmask(raw_image._data)
        if self.orig_mask is np.ma.nomask:
            self.orig_mask = np.zeros(raw_image.shape, np.bool)
        self._min_t = float(np.ma.min(raw_image._data))
        self._max_t = float(np.ma.max(raw_image._data))

        self.overlay = ResampledIndexVolumeSlicer(
            raw_image, bbox=self.bbox,
            norm=(self._min_t, self._max_t)
            )

        # Trigger some properties to clear their caches
        self._recompute_props = True

        # Updating...
        #   scalar norm parameters
        print 'normalizing from', self._min_t, 'to', self._max_t
        self.norm = (self._min_t, self._max_t)
        print self.norm
        #   threshold scalars (just use same array, nothing fancy yet)
        self.threshold.map_scalars = np.asarray(raw_image)
        self.raw_image = raw_image
        if not silently:
            self.send_image_signal()

    def send_image_signal(self):
        print 'sending overlay updated'
        self.overlay_updated = True
        if self.image_signal:
            print 'sending image signal'
            self.image_signal.emit(self)

    def send_location_signal(self, loc):
        self.world_position = np.array(loc)
        self.world_position_updated = True
        if self.loc_signal:
            self.loc_signal.emit(*loc)

    @on_trait_change('norm, cmap_option, interpolation')
    def send_props_signal(self):
        if self.props_signal:
            self.props_signal.emit(self)
    
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
        self.threshold.thresh_map_name = 'overlay scalars'        
        if self.comp == 'greater than':
            self.threshold.thresh_mode = 'mask higher'
            self.threshold.thresh_limits = (self._min_t, self.tval)
        else:
            self.threshold.thresh_mode = 'mask lower'
            self.threshold.thresh_limits = (self.tval, self._max_t)
        self._recompute_props = True
        m_arr = np.ma.masked_array(
            np.asarray(self.raw_image),
            self.mask
            )
        img = ni_api.Image(m_arr, self.raw_image.coordmap)
        self.overlay = ResampledIndexVolumeSlicer(
            img, bbox=self.bbox,
            grid_spacing=self.overlay.grid_spacing,
            norm=self.norm
            )
        self.send_image_signal()

    def _clear_button_fired(self):
        self.threshold.thresh_map_name = ''
        self._recompute_props = True        
        self.send_image_signal()
##         self.send_props_signal()

    @on_trait_change('order') #, dispatch='new')
    def find_peak(self):
        if self.overlay is None:
            return
        if self.ana_xform in ('absmax', 'max'):
            pk_flat_idx = self.ordered_idx[-self.order]
        else:
            pk_flat_idx = self.ordered_idx[self.order-1]

        vol_idx = np.array(np.lib.index_tricks.unravel_index(
            pk_flat_idx, self.raw_image.shape))
        
        xyz_a = self.raw_image.coordmap(vol_idx)
        xyz_b = self.raw_image.coordmap(vol_idx+1)
        pk_val_arr = np.asarray(self.raw_image)[tuple(vol_idx)]
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

    # -- Some ColorbarPanel interaction --
    def connect_colorbar(self, colorbar):
        self.cbar = colorbar
        self._set_cbar_norm()
        self._set_cbar_cmap()
        self._move_cbar_indicator()
    
    @on_trait_change('tval')
    def _move_cbar_indicator(self):
        if not self.cbar:
            return
        self.cbar.change_threshold(self.tval)
    @on_trait_change('norm')
    def _set_cbar_norm(self):
        if not self.cbar:
            return
        print 'image norm changed'
        self.cbar.change_norm(mpl.colors.normalize(*self.norm))
    @on_trait_change('cmap_option')
    def _set_cbar_cmap(self):
        if not self.cbar:
            return
        print 'image cmap changed'
        self.cbar.change_cmap(self.colormap)

    @on_trait_change('grid_size')
    def new_grid_size(self):
        # could also use the overlay.update_grid_spacing function
        img = self.raw_image
##         mask = self.mask
##         new_slicer = SampledVolumeSlicer(img, bbox=self.bbox,
##                                          grid_spacing=[float(self.grid_size)]*3)
        new_slicer = ResampledVolumeSlicer(
            img, bbox=self.bbox, grid_spacing=[float(self.grid_size)]*3
            )
        self.update_overlay(new_slicer)

    # Property Getters
    @cached_property
    def _get_description(self):
        um_pts = np.logical_not(self.mask).sum()
        d_range = self.norm
        dstr = \
"""
Overlay image
data range: (%1.3f, %1.3f)
unmasked points: %d
"""%(d_range[0], d_range[1], um_pts)
        return dstr

    @cached_property
    def _get_mask(self):
        """ Create a negative mask of the overlay map, where points
        masked are marked as True
        """
        if self.overlay is None:
            return None
        m = self.orig_mask.copy() # neg mask
        nm = self.threshold.binary_mask
        if nm is not None:
            m |= nm
        return m

    @cached_property
    def _get_work_arr(self):
        if self.overlay is None:
            return None
        data = np.asarray(self.raw_image)
        return np.ma.masked_array(data, mask=self.mask, copy=False)
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
                Item('comp', label='Mask values'),
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
    overmanager.edit_traits()
