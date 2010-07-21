# NumPy
import numpy as np

# NIPY
from nipy.core import api as ni_api

# Enthought library
from enthought.traits.api import Instance, on_trait_change, Bool, DelegatesTo
from enthought.traits.ui.api import View, Item, Group

# XIPY imports
from xipy.vis.mayavi_widgets import VisualComponent
from xipy.vis.rgba_blending import quick_convert_rgba_to_vtk

class ImageBlendingComponent(VisualComponent):
    """
    A class to take control of the main display's color
    sources for its ImagePlaneWidgets. This class is responsible for
    monitoring changes to the anatomical and functional colors, and
    selecting the appropriate array or blend to plot. This includes
    monitoring the functional manager to control changes to colormapping
    in a BlendedImages object.
    """

    # ----- This will eventually become this VisualComponent's UI widget -----
    show_func = Bool(False)
    show_anat = Bool(False)

    view = View(
        Group(
            Item('show_anat', label='Plot Anatomical'),
            Item('show_func', label='Plot Functional')
            )
        )

    def __init__(self, display, watch_overlay=True, **traits):
        if 'name' not in traits:
            traits['name'] = 'Image Planes'
##         traits['display'] = display
        VisualComponent.__init__(self, **traits)
        self.trait_setq(display=display)
        for trait in ('blender', 'blended_src', 'func_man'):
            self.add_trait(trait, DelegatesTo('display'))

        # -- GUI event, dispatch on new thread
        if watch_overlay:
            # only attach these listeners if the BlendedImages object
            # is not being controlled elsewhere
            self.on_trait_change(self._alpha_scale, 'func_man.alpha_scale',
                                 dispatch='new')
            self.on_trait_change(self._set_over_cmap, 'func_man.cmap_option',
                                 dispatch='new')
            self.on_trait_change(self._set_over_norm, 'func_man.norm',
                                 dispatch='new')

        # -- Data updates
            self.on_trait_change(self._update_colors_from_func_man,
                                 'func_man.overlay_updated')
        # do these in any case
        self.on_trait_change(self._update_colors_from_anatomical,
                             'display.blender.main')
        self.on_trait_change(self._monitor_sources,
                             'blender.main_rgba, blender.over_rgba')

    
    # -- Plotting Toggles ----------------------------------------------------
    @on_trait_change('show_anat')
    def _show_anat(self):
        if not len(self.blender.main_rgba):
            print 'no anat array loaded yet'
            self.set(show_anat=False, trait_change_notify=False)
            return
        # this will assess the status of the image source and plotting
        self.change_blended_source_data()

    @on_trait_change('show_func')
    def _show_func(self):
        if not len(self.blender.over_rgba):
            print 'no overlay present'
            self.set(show_func=False, trait_change_notify=False)
            return
        self.change_blended_source_data()

    # -- Color Mapping Callbacks ---------------------------------------------
    def _alpha_scale(self):
        if not self.func_man:
            return
        self.blender.over_alpha = self.func_man.alpha()

    def _set_over_norm(self):
        print 'resetting scalar normalization from func_man.norm'
        self.blender.over_norm = self.func_man.norm
    
    def _set_over_cmap(self):
        self.blender.over_cmap = self.func_man.colormap

    def _update_colors_from_func_man(self):
        """ When a new overlay is signaled, update the overlay color bytes
        """
        print 'saw func_man overlay_updated fire'
        if not self.func_man or not self.func_man.overlay:
            return
        # this could potentially change scalar mapping properties too
        self.blender.trait_setq(
            over_cmap=self.func_man.colormap,
            over_norm=self.func_man.norm,
            over_alpha=self.func_man.alpha()
            )
            
        self.blender.over = self.func_man.overlay

    def _update_colors_from_anatomical(self):
        if self.blender.main:
            self.change_blended_source_data(new_grid=True)
        else:
            self.show_anat = False
        

    # -- Color Array Monitoring ----------------------------------------------
    def _monitor_sources(self, obj, name, new):
        print name, 'changed'
        if name == 'main_rgba' and self.show_anat:
            self.change_blended_source_data()
        elif name == 'over_rgba' and self.show_func:
            self.change_blended_source_data()
    
    def change_blended_source_data(self, new_grid=False):
        """ Create a pixel-blended array, whose contents depends on the
        current plotting conditions. Also check the status of the
        visibility of the plots.
        """
        if self.show_func and self.show_anat:
            print 'will plot blended'
            img_data = quick_convert_rgba_to_vtk(
                self.blender.blended_rgba
                )
        elif self.show_func:
            print 'will plot over plot'
            img_data = quick_convert_rgba_to_vtk(
                self.blender.over_rgba
                )
        else:
            print 'will plot anatomical'
            img_data = quick_convert_rgba_to_vtk(
                self.blender.main_rgba
                )

        # if a new grid, update (even if invisibly)
        new_grid = new_grid or self.blended_src.scalar_data is None \
                   or self.blended_src.scalar_data.size != img_data.size
        if new_grid:
            print 'changing data not in-place'
            # this will kick off the scalar_data_changed stuff
            self.blended_src.scalar_data = img_data.copy()
            self.blended_src.spacing = self.blender.img_spacing
            self.blended_src.origin = self.blender.img_origin
            self.blended_src.update_image_data = True #???
        
        if not self.show_func and not self.show_anat:
            self.display.toggle_planes_visible(False)
            return

        if not new_grid:
            print 'changing data in-place'
            self.blended_src.scalar_data[:] = img_data

        #self.blended_src.update_image_data = True
        self.blended_src.update()

        if not hasattr(self.display, 'ipw_x'):
            self.display.add_plots_to_scene()
        self.display.toggle_planes_visible(True)

##         if self.show_func and self.show_anat:
##             print 'will plot blended'
##             pscalars = 'blended_colors'
##         elif self.show_func:
##             print 'will plot over plot'
##             pscalars = 'over_colors'
##         else:
##             print 'will plot anatomical'
##             pscalars = 'main_colors'

##         self.ipw_src.point_scalars_name = pscalars
## ##         pdata = self.blended_src.image_data.point_data
## ##         pdata.set_active_attribute(pscalars, 0)
        
##         if not self.show_func and not self.show_anat:
##             self.toggle_planes_visible(False)
##             return

##         self.blended_src.update_image_data = True
## ##         self.blended_src.update()

##         if not hasattr(self, 'ipw_x'):
##             self.add_plots_to_scene()
##         else:            
##             self.toggle_planes_visible(True)
            
