# NumPy
import numpy as np

# NIPY
from nipy.core import api as ni_api

# Enthought library
from enthought.traits.api import HasTraits, Instance, on_trait_change, Array, \
     Bool, Range, Enum, Property, List, Tuple, DelegatesTo, TraitError, String
from enthought.traits.ui.api import View, Item, HGroup, VGroup, Group, \
     RangeEditor
from enthought.tvtk.api import tvtk
from enthought.mayavi.core.api import Source
from enthought.mayavi.sources.array_source import ArraySource
from enthought.mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
     SceneEditor
from enthought.mayavi.modules.text import Text
from enthought.mayavi import mlab

# XIPY imports
from xipy.slicing.image_slicers import ResampledVolumeSlicer, \
     VolumeSlicerInterface
from xipy.overlay import OverlayInterface, ThresholdMap
from xipy.vis.qt4_widgets.auxiliary_window import TopLevelAuxiliaryWindow
from xipy.vis.mayavi_tools import ArraySourceRGBA, image_plane_widget_rgba
from xipy.vis.mayavi_tools import time_wrap as tw
from xipy.vis.mayavi_widgets import VisualComponent
from xipy.vis.rgba_blending import BlendedImages, quick_convert_rgba_to_vtk
import xipy.vis.color_mapping as cm
import xipy.volume_utils as vu

class OverlayThresholdingSurfaceComponent(VisualComponent):
    """A class to take control of thresholding the overlay, and creating
    surfaces of unmasked regions
    """

    show_tsurfs = Bool(False)
    show_csurfs = Bool(False)

    # We will use a BlendedImages instance to keep a native-resolution
    # RGBA image of the functional overlay
    func_blender = Instance(BlendedImages)
    func_src = Instance(Source)

    view = View(
        Group(
            Item('show_tsurfs', label='Display Threshold Surfaces'),
            Item('show_csurfs', label='Display Unmasked Contours')
            )
        )
    # ------------------------------------------------------------------------

    def __init__(self, display, **traits):
        if 'name' not in traits:
            traits['name'] = 'Overlay Surfaces'
        traits['display'] = display
        VisualComponent.__init__(self, **traits)
        for trait in ('func_man', 'master_src'):
            self.add_trait(trait, DelegatesTo('display'))

    def _func_blender_default(self):
        bi = BlendedImages(vtk_order=True)
        return bi

    def _func_src_default(self):
        src = ArraySourceRGBA(transpose_input_array=False)
        src.scalar_name = 'lowres_over_colors'
        return mlab.pipeline.add_dataset(
            src, figure=self.display.scene.mayavi_scene
            )

    @on_trait_change('show_tsurfs')
    def show_thresh_surfaces(self):
        # Turn on/off threshold tracking.. be sneaky and wait until
        # the main BlendedImages over_rgba changes. At that ponit the
        # "over" attribute will be fully resampled
        self.on_trait_change(self._update_mask_channel,
                             'display.blender.over_rgba',
                             remove=not (self.show_tsurfs or self.show_csurfs))
        self.on_trait_change(self._update_overlay_threshold,
                             'display.blender.over_rgba',
                             remove = not self.show_tsurfs)
        if self.show_tsurfs and not self.display.blender.over:
            print 'no overlay thresholding available'
            self.trait_setq(show_tsurfs=False)
            return
        if self.show_tsurfs and \
               (not hasattr(self, 'thresh_surf') or not self.thresh_surf):
            self.add_threshold_surf()
        if hasattr(self, 'thresh_surf'):
            self.thresh_surf.visible = self.show_tsurfs

    @on_trait_change('show_csurfs')
    def show_contour_surfaces(self):
        # Turn on/off threshold tracking.. be sneaky and wait until
        # the main BlendedImages over_rgba changes. At that ponit the
        # "over" attribute will be fully resampled
        self.on_trait_change(self._update_mask_channel,
                             'display.blender.over_rgba',
                             remove=not (self.show_tsurfs or self.show_csurfs))
        self.on_trait_change(self._update_overlay_contour,
                             'display.blender.over_rgba',
                             remove = not self.show_csurfs)
        if self.show_csurfs and not self.display.blender.over:
            print 'no overlay thresholding available'
            self.trait_setq(show_csurfs=False)
            return
        if self.show_csurfs and \
               (not hasattr(self, 'contour_surf') or not self.contour_surf):
            self.add_contour_surf()
        if hasattr(self, 'contour_surf'):
            self.contour_surf.visible = self.show_csurfs
    

    def _update_mask_channel(self):
        bi = self.func_blender

        # just manually sync up the color mapping properties
        mappings = {}
        for trait in ('over_alpha', 'over_cmap', 'over_norm'):
            mappings[trait] = getattr(self.display.blender, trait)
        bi.trait_setq(**mappings)
        
        bi.over = self.display.blender.over
        vtk_arr = quick_convert_rgba_to_vtk(bi.over_rgba)
        self.func_src.spacing = bi.img_spacing
        self.func_src.origin = bi.img_origin
        self.func_src.scalar_data = vtk_arr

        pdata = self.func_src.image_data.point_data
        # The alpha channel in the overlay data should be a good
        # proxy for where the overlay is masked. In any case, wherever
        # alpha is 0, the overlay is invisible in all visualizations
        over_mask = np.clip(bi.over_rgba[...,3], 0, 1)
        mask = pdata.get_array('over_mask')
        if mask:
            mask.from_array(np.ravel(over_mask))
        else:
            n = pdata.add_array(np.ravel(over_mask))
            pdata.get_array(n).name = 'over_mask'
            self.mask_channel = mlab.pipeline.set_active_attribute(
                self.func_src, point_scalars='over_mask'
                )
        self.func_src.update_image_data = True
            
    def _update_overlay_threshold(self):
        if not hasattr(self, 'mask_channel'):
            print 'adding masked channel'
            self._update_mask_channel()
        if not hasattr(self, 'thresh_filter'):
            self.thresh_filter = mlab.pipeline.threshold(self.mask_channel)
            self.thresh_filter.filter_type = 'cells'
            self.threshold_overlay = mlab.pipeline.set_active_attribute(
                self.thresh_filter, point_scalars='lowres_over_colors'
                )
        try:
            self.thresh_filter.lower_threshold = 0.5
        except:
            pass
        try:
            self.thresh_filter.upper_threshold = 1
        except:
            pass
        self.thresh_filter.update_pipeline()

    def _update_overlay_contour(self):
        if not hasattr(self, 'mask_channel'):
            print 'adding masked channel'
            self._update_mask_channel()
        if not hasattr(self, 'contour_filter'):
            self.contour_filter = mlab.pipeline.contour(self.mask_channel)
            normals = mlab.pipeline.poly_data_normals(self.contour_filter)
            normals.filter.feature_angle = 80.
            self.contoured_overlay = mlab.pipeline.set_active_attribute(
                normals, point_scalars='lowres_over_colors'
                )
        self.contour_filter.update_pipeline()
        
    def add_threshold_surf(self):
        if not hasattr(self, 'threshold_overlay'):
            print 'adding threshold channel'
            self._update_overlay_threshold()

        print 'creating surface on threshold channel'
        surf = mlab.pipeline.surface(
            self.threshold_overlay,
            representation='wireframe',
            opacity=0.35, figure=self.display.scene.mayavi_scene
            )
        self.thresh_surf = surf

    def add_contour_surf(self):
        if not hasattr(self, 'contoured_overlay'):
            print 'adding contour channel'
            self._update_overlay_contour()

        print 'creating surface on contour channel'
        surf = mlab.pipeline.surface(
            self.contoured_overlay,
            representation='surface',
            opacity=0.35, figure=self.display.scene.mayavi_scene
            )
        self.contour_surf = surf
