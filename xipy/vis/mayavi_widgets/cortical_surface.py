# NumPy, Scipy
import numpy as np
from scipy import ndimage

# NIPY
from nipy.core import api as ni_api
from nipy.neurospin.utils.emp_null import ENN


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

surf_to_component = {
    'Anatomical' : 'main_colors',
    'Overlay' : 'over_colors',
    'Blended' : 'blended_colors'
    }

component_to_surf = dict( ( (v,u) for u,v in surf_to_component.iteritems() ) )

class CorticalSurfaceComponent(VisualComponent):

    anat_scalars = DelegatesTo('display')
    anat_image = DelegatesTo('display')
    poly_extractor = DelegatesTo('display')
    blender = DelegatesTo('display')
    blended_src = DelegatesTo('display')


    _available_surfaces = Property
    # ----- This will eventually become this VisualComponent's UI widget -----
    show_cortex = DelegatesTo('display')
    surface_component = Enum(values='_available_surfaces')
    # ------------------------------------------------------------------------

    def _get__available_surfaces(self):
        pdata = self.blended_src.image_data.point_data
        anames = [pdata.get_array(n).name
                  for n in xrange(pdata.number_of_arrays)]
        return filter(None, [component_to_surf.get(n, '') for n in names])

    @on_trait_change('show_cortex')
    def _show_cortex(self):
        if not self.blender.main:
            self.trait_setq(show_cortex=False)
            return
        elif not hasattr(self, 'cortical_surf') or not self.cortical_surf:
            self.add_cortical_surf()
        self.cortical_surf.visible = self.show_cortex

    @on_trait_change('surface_component')
    def _change_surf_color(self):
        if not hasattr(self, 'surf_color'):
            return
        point_scalars = surf_to_component[self.surface_component]
        self.surf_color.point_scalars_name = point_scalars
    
    def add_cortical_surf(self):
        # lifted from Gael Varoquax

        # this is fairly brittle-- don't know what will result if
        # the brain is not skull-stripped
        brain_image = self.blender.main.image_arr
        
        enn = ENN(brain_image[(brain_image>0) & (brain_image<256)])
        enn.learn()
        if np.isnan([enn.mu, enn.sigma]).any():
            thresh = 128
        else:
            thresh = enn.mu-2*enn.sigma
        arr = (brain_image > thresh).astype('d')
        arr_blurred = ndimage.gaussian_filter(
            ndimage.binary_fill_holes(
                ndimage.gaussian_filter(arr, 6) > .5
                ).astype('d'),
            2
            )
        arr_blurred *= 255
        arr_blurred = arr_blurred.astype(np.uint8)
        print thresh
        print arr_blurred.any(), arr_blurred.shape
        
##         n = self.blended_src.image_data.point_data.add_array(
##             arr_blurred.T.ravel()
##             )
        # the data from blender should be guaranteed to be in the correct
        # order.....
        n = self.blended_src.image_data.point_data.add_array(
            arr_blurred.ravel()
            )
        self.blended_src.image_data.point_data.get_array(n).name = 'blurred'
        

        anat_blurred = mlab.pipeline.set_active_attribute(
            self.blended_src, point_scalars='blurred'
            )
        
##         anat_blurred.update_pipeline()
        contour = mlab.pipeline.contour(anat_blurred)
        point_scalars = surf_to_component[self.surface_component]
        self.surf_colors = mlab.pipeline.set_active_attribute(
            mlab.pipeline.user_defined(contour,
                                       filter=self.poly_extractor),
            point_scalars=point_scalars
            )
        pnorm = mlab.pipeline.poly_data_normals(self.surf_colors)
        self.cortical_surf = mlab.pipeline.surface(
            pnorm,
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

