# NumPy, Scipy
import numpy as np
from scipy import ndimage

# NIPY
from nipy.core import api as ni_api

# Enthought library
from enthought.traits.api import HasTraits, Instance, on_trait_change, Array, \
     Bool, Range, Enum, Property, List, Tuple, DelegatesTo, TraitError
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
from xipy.vis import rgba_blending
import xipy.volume_utils as vu

class CorticalSurfaceComponent(VisualComponent):

    anat_scalars = DelegatesTo('display')
    anat_image = DelegatesTo('display')
    poly_extractor = DelegatesTo('display')

    # ----- This will eventually become this VisualComponent's UI widget -----
    show_cortex = DelegatesTo('display')
    # ------------------------------------------------------------------------

    @on_trait_change('show_cortex')
    def _show_cortex(self):
        if not self.anat_image:
            self.trait_setq(show_cortex=False)
            return
        elif not hasattr(self, 'cortical_surf') or not self.cortical_surf:
            self.add_cortical_surf()
        self.cortical_surf.visible = self.show_cortex
    
    def add_cortical_surf(self):
        # lifted from Gael Varoquax
        self.display._stop_scene()
        # this seems to
        # 1) smooth a thresholding mask (>4800)
        # 2) fill the holes of wherever this smoothed field is > 0.5
        # 3) smooth the final, filled mask

        # XYZ: TRY TO USE BRAIN MASK EXTRACTION HERE INSTEAD OF THRESHOLD

        # OK.. WANT TO HAVE EXACTLY THE SAME SUPPORT FOR THE SURFACE IMAGE
        # AND THE IMAGE PLANE IMAGES..

##         print 'getting brain mask'
##         img_arr = self.anat_image.image_arr.filled(fill_value=0)
## ##         anat_support = vu.auto_brain_mask(img_arr)
## ##         print 'getting smoothed mask'
## ##         anat_blurred = ndimage.gaussian_filter(anat_support.astype('d'), 6)
## ##         anat_blurred = ( (anat_blurred > .5) | anat_support ).astype('d')
##         anat_blurred = ndimage.gaussian_filter(img_arr, 6)
        
        anat_blurred = ndimage.gaussian_filter(
            (ndimage.morphology.binary_fill_holes(
                ndimage.gaussian_filter(
                    (self.anat_image.image_arr > 4800).astype(np.float), 6)
                > 0.5
                )).astype(np.float),
            2).T.ravel()
        n = self.anat_scalars.image_data.point_data.add_array(
            anat_blurred.T.ravel()
            )
        self.anat_scalars.image_data.point_data.get_array(n).name = 'blurred'
        surf_name = self.anat_scalars.image_data.point_data.get_array(n-1).name
        self.anat_scalars.image_data.point_data.update()
        anat_blurred = mlab.pipeline.set_active_attribute(
            self.anat_scalars, point_scalars='blurred'
            )
        anat_blurred.update_pipeline()
        self._contour = mlab.pipeline.contour(anat_blurred)
        
        csurf = mlab.pipeline.set_active_attribute(
            mlab.pipeline.user_defined(self._contour,
                                       filter=self.poly_extractor),
            point_scalars='scalar'
            )
        self.cortical_surf = mlab.pipeline.surface(
            csurf,
            colormap='copper',
            opacity=1,
            #vmin=4800, vmax=5000)
            vmin=10, vmax=7230,
            figure=self.display.scene.mayavi_scene
            )
        self.cortical_surf.enable_contours = True
        self.cortical_surf.contour.filled_contours = True
        self.cortical_surf.contour.auto_contours = True
##         self.cortical_surf.contour.contours = [5000, 7227.8]
        self.cortical_surf.actor.property.backface_culling = True
        self.cortical_surf.actor.mapper.interpolate_scalars_before_mapping = True
        self.cortical_surf.actor.property.interpolation = 'flat'
        
##         # Add opacity variation to the colormap
##         cmap = self.cortical_surf.module_manager.scalar_lut_manager.lut.table.to_array()
##         cmap[128:, -1] = 0.7*255
##         cmap[:128, -1] = 0.9*255
##         self.cortical_surf.module_manager.scalar_lut_manager.lut.table = cmap
        self.display._start_scene()

