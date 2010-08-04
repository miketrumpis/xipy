# NumPy, Scipy
import numpy as np
from scipy import ndimage

# NIPY
from nipy.neurospin.utils.emp_null import ENN


# Enthought library
from enthought.traits.api import on_trait_change, Bool, Enum, Property, \
     DelegatesTo, cached_property
from enthought.traits.ui.api import View, Item, HGroup, Group, Label
from enthought.mayavi import mlab
from enthought.mayavi.sources.array_source import ArraySource
from enthought.tvtk.api import tvtk

# XIPY imports
from xipy.vis.mayavi_widgets import VisualComponent
from xipy.colors.mayavi_tools import MasterSource

surf_to_component = {
    'Anatomical' : MasterSource.main_channel,
    'Overlay' : MasterSource.over_channel,
    'Blended' : MasterSource.blended_channel
    }

component_to_surf = dict( ( (v,u) for u,v in surf_to_component.iteritems() ) )

class CorticalSurfaceComponent(VisualComponent):

    _available_surfaces = Property(
        depends_on='display.blender.over, display.blender.main')
    show_cortex = Bool(False)
    surface_component = Enum(values='_available_surfaces')
    cutout = Bool(False)
    view = View(
        HGroup(
            Group(
                Item('show_cortex', label='Show Cortical Surface'),
                Item('surface_component', label='Surface Colors'),
                Item('cutout', label='Cut-out Mode',
                     enabled_when='object.show_cortex')
                ),
            Group(
                Label('Caution! Are you sure that this image is skull '\
                      'stripped using BET?')
                )
            )
        )
    # ------------------------------------------------------------------------
    def __init__(self, display, **traits):
        if 'name' not in traits:
            traits['name'] = 'Cortical Surface'
        traits['display'] = display
        VisualComponent.__init__(self, **traits)
        for trait in ('poly_extractor', 'master_src',
                      '_planes_function', '_volume_function'):
            self.add_trait(trait, DelegatesTo('display'))
        if not self._volume_function.volume:
            self._volume_function.volume = self.master_src.data
        self.poly_extractor.implicit_function = self._volume_function

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
##         self.cortical_surf.visible = self.show_cortex
        if not self.show_cortex:
            self.bcontour.stop()
        else:
            self.bcontour.start()
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

        # since this is skull stripped from BET, the boundary will be
        # wherever the image drops off to 0
        arr = ndimage.gaussian_filter(
            (brain_image > 0).astype('d'), 6
            )
        mask = ndimage.binary_fill_holes(arr > .5)
        # iterations x voxel size ~= erosion depth??
        mask = ndimage.binary_erosion(mask, iterations=5)
        
        arr_blurred = ndimage.gaussian_filter(
            mask.astype('d'), 2
            )

        # This AA filter will rather dangle off the pipeline..
        # it will be the "source" of the ProbeFilter, but we want
        # to keep it pipeline-enabled so that we can dynamically
        # change colors
        point_scalars = surf_to_component[self.surface_component]
        surf_colors = mlab.pipeline.set_active_attribute(
            self.master_src, point_scalars=point_scalars
            )

        # Now, make a new ArraySource with the attributes copied
        # from master_src
        anat_blurred = ArraySource(transpose_input_array=False)
        anat_blurred.scalar_data = arr_blurred
        anat_blurred.scalar_name = 'blurred'
        # XXX: TRANSLATING HACK!!!!
##         anat_blurred.origin = self.master_src.data.origin
        anat_blurred.spacing = self.master_src.data.spacing
        anat_blurred = mlab.pipeline.add_dataset(anat_blurred)
        
##         anat_blurred.update_pipeline()
        contour = mlab.pipeline.contour(anat_blurred)
        decimated = mlab.pipeline.decimate_pro(contour)
        extracted = mlab.pipeline.user_defined(
            decimated, filter=self.poly_extractor
            )

        sampler = tvtk.ProbeFilter()
        sampler.source = surf_colors.outputs[0]

        surf_points = mlab.pipeline.user_defined(extracted, filter=sampler)

        self.cortical_surf = mlab.pipeline.surface(
            surf_points,
            opacity=.95
            )
        self.cortical_surf.actor.property.backface_culling = True
        self.bcontour = contour
        self.surf_colors = surf_colors
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

    @on_trait_change('cutout')
    def _toggle_poly_extractor_mode(self, obj, name, cut_mode):
        if cut_mode:
            self.poly_extractor.implicit_function = self._planes_function
        else:
            if not self._volume_function.volume:
                self._volume_function.volume = self.master_src.data
            self.poly_extractor.implicit_function = self._volume_function
        
##         self.poly_extractor.extract_inside = not cut_mode
        self.poly_extractor.extract_inside = False #???
        self.poly_extractor.update()
