# NumPy
import numpy as np

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
from xipy.vis.rgba_blending import quick_convert_rgba_to_vtk
import xipy.volume_utils as vu

class OverlayBlendingComponent(VisualComponent):
    """A class to take control of blending overlay colors into the main
    display's blended image source
    """
    
    blender = DelegatesTo('display')
    blended_src = DelegatesTo('display')    
    func_man = DelegatesTo('display')

    # ----- This will eventually become this VisualComponent's UI widget -----
    show_func = DelegatesTo('display')
    # ------------------------------------------------------------------------
    @on_trait_change('show_func')
    def _show_func(self):
        if not self.func_man or not self.func_man.overlay:
            print 'no functional manager to provide an overlay'
            self.set(show_func=False, trait_change_notify=False)
            return
        self.display.change_source_data()
    
    @on_trait_change('func_man.alpha_scale')
    def _alpha_scale(self):
        if not self.func_man:
            return
        self.blender.over_alpha = self.func_man.alpha()

    @on_trait_change('func_man.norm')
    def _set_blender_norm(self):
        print 'resetting scalar normalization from func_man.norm'
        self.blender.over_norm = self.func_man.norm
    
    @on_trait_change('func_man.cmap_option')
    def _set_over_cmap(self):
        self.blender.over_cmap = self.func_man.colormap

    @on_trait_change('func_man.overlay_updated')
    def _update_colors_from_func_man(self):
        """ When a new overlay is signaled, update the overlay color bytes
        """
        if not self.func_man or not self.func_man.overlay:
            return
        # this could potentially change scalar mapping properties too
        self.blender.trait_setq(
            over_cmap=self.func_man.colormap,
            over_norm=self.func_man.norm,
            over_alpha=self.func_man.alpha()
            )
            
        self.blender.over = self.func_man.overlay
        idata = self.blended_src.image_data
        vtk_colors = quick_convert_rgba_to_vtk(self.blender.over_rgba)
        vtk_colors.shape = ( np.prod(vtk_colors.shape[:3]), 4 )
        colors = idata.point_data.get_array('over_colors')
        if colors:
            colors.from_array(vtk_colors)
        else:
            n = idata.point_data.add_array(vtk_colors)
            idata.point_data.get_array(n).name = 'over_colors'

        vtk_colors = quick_convert_rgba_to_vtk(self.blender.blended_rgba)
        vtk_colors.shape = ( np.prod(vtk_colors.shape[:3]), 4 )        
        colors = idata.point_data.get_array('blended_colors')
        if colors:
           colors.from_array(vtk_colors)
        else:
            n = idata.point_data.add_array(vtk_colors)
            idata.point_data.get_array(n).name = 'blended_colors'

        self.blended_src.update_image_data = True
##         self.display.ipw_src.update_pipeline()
            
