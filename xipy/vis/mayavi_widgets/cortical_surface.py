# NumPy, Scipy
import numpy as np
from scipy import ndimage

# NIPY
from nipy.core import api as ni_api
from nipy.neurospin.utils.emp_null import ENN


# Enthought library
from enthought.traits.api import HasTraits, Instance, on_trait_change, Array, \
     Bool, Range, Enum, Property, List, DelegatesTo, TraitError, String, \
     cached_property
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
from xipy.vis.mayavi_widgets import VisualComponent, MasterSource
import xipy.volume_utils as vu

surf_to_component = {
    'Anatomical' : MasterSource.main_channel,
    'Overlay' : MasterSource.over_channel,
    'Blended' : MasterSource.blended_channel
    }

component_to_surf = dict( ( (v,u) for u,v in surf_to_component.iteritems() ) )

class CorticalSurfaceComponent(VisualComponent):

    _available_surfaces = Property(depends_on='display.blender.over')
    # ----- This will eventually become this VisualComponent's UI widget -----
    show_cortex = Bool(False)
    surface_component = Enum(values='_available_surfaces')
    view = View(
        Group(
            Item('show_cortex', label='Show Cortical Surface'),
            Item('surface_component', label='Surface Colors')
            )
        )
    # ------------------------------------------------------------------------
    def __init__(self, display, **traits):
        if 'name' not in traits:
            traits['name'] = 'Cortical Surface'
        traits['display'] = display
        VisualComponent.__init__(self, **traits)
        for trait in ('poly_extractor', 'master_src'):
            self.add_trait(trait, DelegatesTo('display'))

    @cached_property
    def _get__available_surfaces(self):
        return [component_to_surf[ch] for ch in self.master_src.rgba_channels]

    @on_trait_change('show_cortex')
    def _show_cortex(self):
        if not self.master_src.blender.main:
            self.trait_setq(show_cortex=False)
            return
        elif not hasattr(self, 'cortical_surf') or not self.cortical_surf:
            self.add_cortical_surf()
        self.cortical_surf.visible = self.show_cortex

    @on_trait_change('surface_component')
    def _change_surf_color(self):
        if not hasattr(self, 'surf_colors'):
            return
        point_scalars = surf_to_component[self.surface_component]
        self.surf_colors.point_scalars_name = point_scalars
    
    def add_cortical_surf(self):
        # lifted from Gael Varoquax

        # this is fairly brittle-- don't know what will result if
        # the brain is not skull-stripped

        # brain_image is (currently) a copy of the integer indices
        # (not the "real valued" scalars)
        brain_image = self.master_src.blender.main.image_arr.copy()
        np.putmask(brain_image, brain_image>255, 0)
        
        arr = ndimage.gaussian_filter(
            (brain_image > 0).astype('d'), 6
            )
        mask = ndimage.binary_fill_holes(arr > .5)
        # iterations x voxel size ~= erosion depth??
        mask = ndimage.binary_erosion(mask, iterations=5)
        
        arr_blurred = ndimage.gaussian_filter(
            mask.astype('d'), 2
            )
        arr_blurred *= 255
        np.clip(arr_blurred, 0, 255, out=arr_blurred)
        arr_blurred = arr_blurred.astype(np.uint8)
        
        # the data from blender should be guaranteed to be in the correct
        # order.....
        n = self.master_src.image_data.point_data.add_array(
            np.ravel(arr_blurred)
            )
        self.master_src.image_data.point_data.get_array(n).name = 'blurred'
        

        anat_blurred = mlab.pipeline.set_active_attribute(
            self.master_src, point_scalars='blurred'
            )
        
##         anat_blurred.update_pipeline()
        contour = mlab.pipeline.contour(anat_blurred)
        decimate = mlab.pipeline.decimate_pro(contour)
        extracted = mlab.pipeline.user_defined(
            decimate, filter=self.poly_extractor
            )
        point_scalars = surf_to_component[self.surface_component]
        self.surf_colors = mlab.pipeline.set_active_attribute(
            extracted,
            point_scalars=point_scalars
            )
##         pnorm = mlab.pipeline.poly_data_normals(self.surf_colors)
        self.cortical_surf = mlab.pipeline.surface(
            self.surf_colors,
            opacity=.95,
            figure=self.display.scene.mayavi_scene
            )
        self.cortical_surf.actor.property.backface_culling = True

##         self.cortical_surf.enable_contours = True
##         self.cortical_surf.contour.filled_contours = True
##         self.cortical_surf.contour.auto_contours = True
## ##         self.cortical_surf.contour.contours = [5000, 7227.8]
##         self.cortical_surf.actor.mapper.interpolate_scalars_before_mapping = True
##         self.cortical_surf.actor.property.interpolation = 'flat'
        
##         # Add opacity variation to the colormap
##         cmap = self.cortical_surf.module_manager.scalar_lut_manager.lut.table.to_array()
##         cmap[128:, -1] = 0.7*255
##         cmap[:128, -1] = 0.9*255
##         self.cortical_surf.module_manager.scalar_lut_manager.lut.table = cmap

