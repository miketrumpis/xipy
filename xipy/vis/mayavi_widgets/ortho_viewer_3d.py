# NumPy
import numpy as np

# NIPY
from nipy.core import api as ni_api

# Enthought library
from enthought.traits.api import HasTraits, Instance, on_trait_change, \
     Bool, List
from enthought.traits.ui.api import View, Item, HGroup, VGroup, Group, \
     ListEditor
from enthought.tvtk.api import tvtk
from enthought.mayavi.core.api import Source, Filter
from enthought.mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
     SceneEditor
from enthought.mayavi.modules.text import Text
from enthought.mayavi import mlab

# XIPY imports
from xipy.slicing.image_slicers import VolumeSlicerInterface
from xipy.overlay.interface import OverlayInterface
from xipy.colors.mayavi_tools import ArraySourceRGBA, image_plane_widget_rgba, \
     set_image_active_attribute
from xipy.colors.rgba_blending import BlendedImages
import xipy.volume_utils as vu

from xipy.vis.mayavi_widgets import VisualComponent
from xipy.colors.mayavi_tools import MasterSource, disable_render

def three_plane_pt(n1, n2, n3, x1, x2, x3):
    nm = np.array((n1,n2,n3)).T
    dt = np.linalg.det(nm)
    n2x3 = np.cross(n2,n3)
    n3x1 = np.cross(n3,n1)
    n1x2 = np.cross(n1,n2)
    x = ( np.dot(x1,n1)*n2x3 + np.dot(x2,n2)*n3x1 + np.dot(x3,n3)*n1x2 )
    return x / dt


class OrthoViewer3D(HasTraits):
    #---------------------------------------------------------------------------
    # Data and Figure
    #---------------------------------------------------------------------------
    scene = Instance(MlabSceneModel, ())

    blender = Instance(BlendedImages)

##     blended_src = Instance(Source)
    master_src = Instance(Source)

    anat_image = Instance(VolumeSlicerInterface)

    #---------------------------------------------------------------------------
    # Functional Overlay Manager
    #---------------------------------------------------------------------------
    func_man = Instance(OverlayInterface, ())

    #---------------------------------------------------------------------------
    # Our ImagePlaneWidget color filter
    #---------------------------------------------------------------------------
    principle_plane_colors = Instance(Filter)

    #---------------------------------------------------------------------------
    # VisualComponents List
    #---------------------------------------------------------------------------
    vis_helpers = List(VisualComponent)

    #---------------------------------------------------------------------------
    # Other Traits
    #---------------------------------------------------------------------------
    poly_extractor = Instance(tvtk.ExtractPolyDataGeometry, ())
    _planes_function = Instance(
        tvtk.Planes, args=(),
        kw=dict(points=[ (0,0,0), (0,0,0), (0,0,0) ],
                normals=[ (-1,0,0), (0,-1,0), (0,0,-1) ])
        )
    _volume_function = Instance(tvtk.ImplicitVolume, ())
    info = Instance(Text)
    
    _axis_index = dict(x=0, y=1, z=2)

    def __init__(self, parent=None, manage_overlay=True, **traits):
        HasTraits.__init__(self, **traits)
        self.master_src # instantiates default master_src and blender
        self.func_man
        from xipy.vis.mayavi_widgets.overlay_blending import ImageBlendingComponent
        from xipy.vis.mayavi_widgets.overlay_thresholding_surface import OverlayThresholdingSurfaceComponent
        from xipy.vis.mayavi_widgets.cortical_surface import CorticalSurfaceComponent
        from xipy.vis.mayavi_widgets.translated_planes import TranslatedPlanes

        # set up components
        self.vis_helpers.append(
            ImageBlendingComponent(self, manage_overlay)
            )
        self.vis_helpers.append(
            TranslatedPlanes(self)
            )
        self.vis_helpers.append(
            OverlayThresholdingSurfaceComponent(self)
            )

        self.vis_helpers.append(
            CorticalSurfaceComponent(self)
            )
                        
        
        anat_alpha = np.ones(256)
        anat_alpha[:5] = 0
        self.blender.set(main_alpha=anat_alpha, trait_notify_change=False)
        self.__reposition_planes_after_interaction = False

    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _blender_default(self):
        return BlendedImages(vtk_order=True)
##     def _blended_src_default(self):
##         s = ArraySourceRGBA(transpose_input_array=False)
##         # by default this array has a 'main_colors' array..
##         # later, it may have 'over_colors' and 'blended_colors'
##         s.scalar_name = 'ipw_colors'
##         b_src = mlab.pipeline.add_dataset(s, figure=self.scene.mayavi_scene)
##         return b_src
    def _master_src_default(self):
        s = MasterSource(blender = self.blender)
        # make sure that the image blender is sync'd up
        self.sync_trait('blender', s, mutual=False)
        return mlab.pipeline.add_dataset(s, figure=self.scene.mayavi_scene)
    def _principle_plane_colors_default(self):
        return set_image_active_attribute(self.master_src)
##         return mlab.pipeline.set_active_attribute(self.master_src)
    def _func_man_default(self):
        return OverlayInterface()
    def _info_default(self):
        info = mlab.text(.05,.05,'Welcome',width=0.4,
                         figure=self.scene.mayavi_scene)
        info.property.font_size = 10
        return info

    #---------------------------------------------------------------------------
    # Construct ImplicitPlaneWidget plots
    #---------------------------------------------------------------------------
    def make_ipw(self, axis_name):
        self.master_src.pipeline_changed = True
        ipw = image_plane_widget_rgba(
            self.principle_plane_colors,
            figure=self.scene.mayavi_scene,
            plane_orientation='%s_axes'%axis_name
            )
        ipw.use_lookup_table = False
        ipw.ipw.reslice_interpolate = 0 # hmmm?
        return ipw

    @disable_render
    def add_plots_to_scene(self):
        data_shape = self.master_src.data.dimensions
        p_idx = [data_shape[d]/4. for d in [0,1,2]]
        p_vox = self.blender.coordmap(p_idx)
        print 'initial vox position:', p_vox
        for ax in ('x', 'y', 'z'):
            print self.principle_plane_colors.outputs[0].origin
            ipw = self.make_ipw(ax)
            print self.principle_plane_colors.outputs[0].origin
            # position the image plane widget at an unobtrusive cut
            dim = self._axis_index[ax]
            ipw.ipw.slice_position = p_vox[dim]
            # switch actions here
            ipw.ipw.middle_button_action = 2
            ipw.ipw.right_button_action = 0
            # name this widget and attribute it to self
            setattr(self, 'ipw_%s'%ax, ipw)
            ipw.ipw.add_observer(
                'StartInteractionEvent',
                getattr(self, '_%s_plane_interaction'%ax)
                )
            ipw.ipw.add_observer(
                'InteractionEvent',
                getattr(self, '_%s_plane_interaction'%ax)
                )
            ipw.ipw.add_observer(
                'EndInteractionEvent',
                self._handle_end_interaction
                )


    def _ipw_x(self, axname):
        return getattr(self, 'ipw_%s'%axname, None)
    
    #---------------------------------------------------------------------------
    # Scene interaction callbacks 
    #---------------------------------------------------------------------------
    def _x_plane_interaction(self, widget, event):
        self._handle_plane_interaction('x', widget)
    def _y_plane_interaction(self, widget, event):
        self._handle_plane_interaction('y', widget)
    def _z_plane_interaction(self, widget, event):
        self._handle_plane_interaction('z', widget)
    def _handle_plane_interaction(self, ax, widget):
        if widget.GetCursorDataStatus():
            # In other words, if moving the crosshairs in a plane
            print 'listening for endinteraction'
            self.__reposition_planes_after_interaction = True
            return
        # otherwise, do normal interaction
        self._link_plane_points(ax, widget)

    def _link_plane_points(self, ax, widget):
        ax_idx = self._axis_index[ax]
        ipw = self._ipw_x(ax).ipw #getattr(self, 'ipw_%s'%ax).ipw
        planes = self._planes_function
        planes.points[ax_idx] = ipw.center
        planes.normals[ax_idx] = map(lambda x: -x, ipw.normal)

    def _handle_end_interaction(self, widget, event):
        if self.__reposition_planes_after_interaction:
            pos_ijk = widget.GetCurrentCursorPosition()
            pos = self.blender.coordmap(pos_ijk[::-1])
            self._snap_to_position(pos)
            self.__reposition_planes_after_interaction = False
        else:
            pos = self._current_intersection()
        self._register_position(pos)

    def _current_intersection(self):
        nx = self.ipw_x.ipw.normal; cx = self.ipw_x.ipw.center
        ny = self.ipw_y.ipw.normal; cy = self.ipw_y.ipw.center
        nz = self.ipw_z.ipw.normal; cz = self.ipw_z.ipw.center
        return three_plane_pt(nx, ny, nz, cx, cy, cz)

    @disable_render
    def _snap_to_position(self, pos):
        if self._ipw_x('x') is None:
            return
        print 'snapping to', pos
##         self._stop_scene()
        anames = ('x', 'y', 'z')
        pd = dict(zip( anames, pos ))
        for ax in anames:
            ipw = self._ipw_x(ax).ipw
            ipw.plane_orientation='%s_axes'%ax
            ipw.slice_position = pd[ax]
##         self._start_scene()

    def _register_position(self, pos):
        print 'registering pos', pos
        if self.func_man:
            self.func_man.world_position = pos
##         self.master_src.update()        
    #---------------------------------------------------------------------------
    # Traits callbacks
    #---------------------------------------------------------------------------

    @on_trait_change('func_man.world_position_updated')
    def _follow_functional_position(self):
        self._snap_to_position(self.func_man.world_position)
    
    @on_trait_change('func_man, func_man.overlay_updated')
    def _update_functional_info(self):
        """ Update misc. aspect of the display when a new overlay crops up
        """
        if not self.func_man or not self.func_man.overlay:
            return
        if self.func_man.description:
            self.info.text = self.func_man.description

    #---------------------------------------------------------------------------
    # Scene update methods
    #---------------------------------------------------------------------------
##     def _stop_scene(self):
##         self._restore_render_mode = self.scene.disable_render
##         self.scene.disable_render = True
##     def _start_scene(self):
##         # by default, restore disable_render mode to False
##         prev_mode = getattr(self, '_restore_render_mode', False)
##         self.scene.disable_render = prev_mode
##         if not prev_mode:
##             self.scene.render_window.render()

    def toggle_planes_visible(self, value):
##         self._toggle_poly_extractor_mode(cut_mode=value)
        for ax in ('x', 'y', 'z'):
            ipw = self._ipw_x(ax) #getattr(self, 'ipw_%s'%ax, False)
            if ipw:
                ipw.visible = value
            
    #---------------------------------------------------------------------------
    # Scene activation callbacks 
    #---------------------------------------------------------------------------
    @on_trait_change('scene.activated')
    def display_scene3d(self):
        self.scene.mlab.view(100, 100)
        self.scene.renderer.use_depth_peeling = True
##         self.scene.scene.background = (0, 0, 0)

        # Keep the view always pointing up
        self.scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
        self.scene.picker.pointpicker.add_observer('EndPickEvent',
                                                   self.pick_callback)
        self.scene.picker.show_gui = False
        self.scene.disable_render = False
        self.scene.camera.position = 500, 300, 300


    def pick_callback(self, picker_obj, evt):
        p = tvtk.to_tvtk(picker_obj)
        x, y, z = p.pick_position
        self._snap_to_position((x,y,z))
        self._register_position((x,y,z))
        
    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(
        VGroup(
            HGroup(
                Item('scene',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=600, width=900),
                show_labels=False,
                dock='vertical',
                springy=True
                ),
            HGroup(
                Item('vis_helpers', style='custom',
                     editor=ListEditor(use_notebook=True,
                                       deletable=False,
                                       page_name='.name')
                     )
                ),
            ),
        resizable=True,
        title='XIPY 3D Viewer Controls',
        )

