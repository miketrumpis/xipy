# NumPy
import numpy as np

# NIPY
from nipy.core import api as ni_api

# Enthought library
from enthought.traits.api import Instance, on_trait_change, Bool, DelegatesTo
from enthought.traits.ui.api import View, Item, Group

# XIPY imports
from xipy.vis.mayavi_widgets import VisualComponent
from xipy.colors.rgba_blending import quick_convert_rgba_to_vtk

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
        VisualComponent.__init__(self, **traits)
        self.trait_setq(display=display)
        for trait in ('blender', 'func_man', 'master_src',
                      'principle_plane_colors'):
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

        self.on_trait_change(self.change_colors,
                             'master_src.colors_changed')

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

    # -- Plotting Toggles ----------------------------------------------------
    # And
    # -- Color Array Monitoring ----------------------------------------------
    @on_trait_change('show_func, show_anat')
    def change_colors(self):
        """
        Change the active color channel on the principle_plane_colors (if
        the channel is available)
        """
        main_chan = self.master_src.main_channel
        over_chan = self.master_src.over_channel
        blnd_chan = self.master_src.blended_channel
        all_rgba = self.master_src.rgba_channels
        if self.show_func and self.show_anat:
            if blnd_chan not in all_rgba:
                if main_chan not in all_rgba:
                    self.show_anat = False
                    return
                if over_chan not in all_rgba:
                    self.show_func = False
                    return
            print 'will plot blended'
            color = blnd_chan
        elif self.show_func:
            if over_chan not in all_rgba:
                self.show_func = False
                return
            print 'will plot over plot'
            color = over_chan
        elif self.show_anat:
            if main_chan not in all_rgba:
                self.show_anat = False
                return
            print 'will plot anatomical'
            color = main_chan
        else:
            print 'turning off plots'
            color = ''

        print 'changing'
        self.principle_plane_colors.point_scalars_name = color
        print 'done'

        if not color:
            self.display.toggle_planes_visible(False)
            return

        ipwx = self.display._ipw_x('x')
        if not ipwx:
            self.display.add_plots_to_scene()
        elif not ipwx.visible:
            self.display.toggle_planes_visible(True)

        self.display.scene.render()
 
            
