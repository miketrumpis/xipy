import os
os.environ['ETS_TOOLKIT'] = 'qt4'

from PyQt4 import QtGui, QtCore

from enthought.traits.api import HasTraits, Instance, on_trait_change, Array, \
     Bool, Range, Enum, Property, List, Tuple
from enthought.traits.ui.api import View, Item, HGroup, VGroup, Group, \
     RangeEditor
from enthought.tvtk.api import tvtk
from enthought.mayavi.core.api import Source
from enthought.mayavi.sources.array_source import ArraySource
from enthought.mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
     SceneEditor
from enthought.mayavi.modules.text import Text
from enthought.mayavi import mlab

import numpy as np
from scipy import ndimage
## from matplotlib import cm
import xipy.vis.color_mapping as cm

from xipy.slicing.image_slicers import ResampledVolumeSlicer, \
     VolumeSlicerInterface
from xipy.overlay import OverlayInterface
from xipy.vis.qt4_widgets.auxiliary_window import TopLevelAuxiliaryWindow
from xipy.vis.mayavi_tools import ArraySourceRGBA, image_plane_widget_rgba
from xipy.vis.mayavi_tools import time_wrap as tw
from xipy.vis import rgba_blending
import xipy.volume_utils as vu
from nipy.core import api as ni_api

import time

import cProfile, pstats

P_THRESH = 0.05
CLUSTER_THRESH = 3

def three_plane_pt(n1, n2, n3, x1, x2, x3):
    nm = np.array((n1,n2,n3)).T
    dt = np.linalg.det(nm)
    n2x3 = np.cross(n2,n3)
    n3x1 = np.cross(n3,n1)
    n1x2 = np.cross(n1,n2)
    x = ( np.dot(x1,n1)*n2x3 + np.dot(x2,n2)*n3x1 + np.dot(x3,n3)*n1x2 )
    return x / dt


class OrthoView3D(HasTraits):
    #---------------------------------------------------------------------------
    # Data and Figure
    #---------------------------------------------------------------------------
    scene = Instance(MlabSceneModel, ())

    anat_src = Instance(Source)

    blender = Instance(rgba_blending.BlendedImage, (),
                       spline_order=0, transpose_inputs=False)
    blended_src = Instance(Source)
##     _blended_src_scalars = Array(dtype='B')
    blob_surf_src = Instance(Source)

    anat_image = Instance(VolumeSlicerInterface)

    #---------------------------------------------------------------------------
    # Functional Overlay Manager
    #---------------------------------------------------------------------------
    func_man = Instance(OverlayInterface)

    #---------------------------------------------------------------------------
    # Scene Control Traits
    #---------------------------------------------------------------------------
    _pscores = List
    pscore_map = Enum(values='_pscores')
    cluster_threshold = Range(low=1,high=30, value=6)
    sig_threshold = Range(low=0.0, high=1.0, value=0.05,
                          editor=RangeEditor(low=0, high=1.0, format='%1.3f'))
    

    show_psurfs = Bool(False)
    show_anat = Bool(False)
    show_func = Bool(False)
    show_cortex = Bool(False)
    alpha_compress = Range(low=0.0, high=4.0, value=1.0,
                           editor=RangeEditor(low=0.0, high=4.0,
                                              format='%1.2f', mode='slider'))


    #---------------------------------------------------------------------------
    # Other Traits
    #---------------------------------------------------------------------------
    #planes_function = Instance(tvtk.Planes, ())
    poly_extractor = Instance(tvtk.ExtractPolyDataGeometry, ())
    info = Instance(Text)
    
    _axis_index = dict(x=0, y=1, z=2)

    def __init__(self, parent=None, **traits):
        HasTraits.__init__(self, **traits)
        anat_alpha = np.ones(256)
        anat_alpha[:5] = 0
        self.blender.set(main_alpha=anat_alpha, trait_notify_change=False)
        self.__reposition_planes_after_interaction = False

    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _anat_src_default(self):
        s = ArraySource(transpose_input_array=False)
        return mlab.pipeline.add_dataset(s, figure=self.scene.mayavi_scene)

    def _blended_src_default(self):
        s = ArraySourceRGBA(transpose_input_array=False)
        return mlab.pipeline.add_dataset(s, figure=self.scene.mayavi_scene)
    
    def _blob_surf_src_default(self):
        return mlab.pipeline.scalar_field(np.zeros((2,2,2)),
                                          figure=self.scene.mayavi_scene)
    
##     #---------------------------------------------------------------------------
##     # Property Getters
##     #---------------------------------------------------------------------------
##     def _get__pscores(self):
##         if not self.func_man:
##             return []
##         return filter(lambda x: x.lower().find('p val')>=0,
##                       self.func_man._stats_maps)
    
    #---------------------------------------------------------------------------
    # Construct ImplicitPlaneWidget plots
    #---------------------------------------------------------------------------
    def make_ipw(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(
            self.blended_src,
            figure=self.scene.mayavi_scene,
            plane_orientation='%s_axes'%axis_name
            )
        ipw.use_lookup_table = False
        ipw.ipw.reslice_interpolate = 0 # hmmm?
        return ipw
    
    def add_plots_to_scene(self):
        self._stop_scene()
        for ax in ('x', 'y', 'z'):
            ipw = self.make_ipw(ax)

            # position the image plane widget at an unobtrusive cut
            dim = self._axis_index[ax]
            data_shape = self.blended_src.image_data.extent[::2]
            ipw.ipw.slice_position = data_shape[dim]/4
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
                self._register_position
                )

        self._start_scene()

    def _ipw_x(self, axname):
        return getattr(self, 'ipw_%s'%axname, None)

    def _snap_to_position(self, pos):
        self._stop_scene()
        anames = ('x', 'y', 'z')
        pd = dict(zip( anames, pos ))
        for ax in anames:
            ipw = self._ipw_x(ax).ipw
            ipw.plane_orientation='%s_axes'%ax
            ipw.slice_position = pd[ax]
        self._start_scene()
        
        
        
    def add_cortical_surf(self):
        # lifted from Gael Varoquax
        self._stop_scene()
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
        n = self.anat_src.image_data.point_data.add_array(
            anat_blurred.T.ravel()
            )
        self.anat_src.image_data.point_data.get_array(n).name = 'blurred'
        surf_name = self.anat_src.image_data.point_data.get_array(n-1).name
        self.anat_src.image_data.point_data.update()
        anat_blurred = mlab.pipeline.set_active_attribute(
            self.anat_src, point_scalars='blurred'
            )
        anat_blurred.update_pipeline()
        self._contour = mlab.pipeline.contour(anat_blurred)
        
        csurf = mlab.pipeline.set_active_attribute(
            mlab.pipeline.user_defined(self._contour,
                                       filter=self.poly_extractor),
            point_scalars='scalar'
            )
        self.cortical_surf = mlab.pipeline.surface(csurf,
                                                   colormap='copper',
                                                   opacity=1,
                                                   #vmin=4800, vmax=5000)
                                                   vmin=10, vmax=7230)
        self.cortical_surf.enable_contours = True
        self.cortical_surf.contour.filled_contours = True
        self.cortical_surf.contour.auto_contours = True
##         self.cortical_surf.contour.contours = [5000, 7227.8]
##         self.cortical_surf.actor.property.frontface_culling = True
        self.cortical_surf.actor.mapper.interpolate_scalars_before_mapping = True
        self.cortical_surf.actor.property.interpolation = 'flat'
        
##         # Add opacity variation to the colormap
##         cmap = self.cortical_surf.module_manager.scalar_lut_manager.lut.table.to_array()
##         cmap[128:, -1] = 0.7*255
##         cmap[:128, -1] = 0.9*255
##         self.cortical_surf.module_manager.scalar_lut_manager.lut.table = cmap
        self._start_scene()
    
    def add_significance_surfs(self):
        self._stop_scene()
        
        tfilter = tvtk.ImageThreshold()
        tfilter.threshold_between(0, .1)
        p_thresh = mlab.pipeline.user_defined(self.blob_surf_src,
                                              filter=tfilter)
        pcontour = mlab.pipeline.contour(p_thresh)
        pcontour.filter.contours = [.02, 0.04, 0.051]
        smooth_ = tvtk.SmoothPolyDataFilter(
            number_of_iterations=10,
            relaxation_factor=0.1,
            feature_angle=60,
            feature_edge_smoothing=False,
            boundary_smoothing=False,
            convergence=0.,
            )
        smooth_contour = mlab.pipeline.user_defined(pcontour, filter=smooth_)
        self.pval_surfs = mlab.pipeline.surface(smooth_contour,
                                                opacity=.35,
                                                colormap='hot',
                                                vmin=0, vmax=P_THRESH*1.5)
        mm = self.pval_surfs.parent
        mm.scalar_lut_manager.reverse_lut = True
        self._start_scene()

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
        ipw = getattr(self, 'ipw_%s'%ax).ipw
        planes_function = self.poly_extractor.implicit_function
        planes_function.points[ax_idx] = ipw.center
        planes_function.normals[ax_idx] = map(lambda x: -x, ipw.normal)

    def _register_position(self, widget, event):
        if self.__reposition_planes_after_interaction:
            pos_ijk = widget.GetCurrentCursorPosition()
            pos = self.anat_image.coordmap(pos_ijk)[0]
            print 'snapping to pos', pos
            self._snap_to_position(pos)
            self.__reposition_planes_after_interaction = False
        else:
            pos = self._current_intersection()
        if self.func_man:
            self.func_man.world_position = self._current_intersection()
        if hasattr(self, 'cortical_surf') and self.show_cortex:
            self.anat_src.update()

    def _current_intersection(self):
        nx = self.ipw_x.ipw.normal; cx = self.ipw_x.ipw.center
        ny = self.ipw_y.ipw.normal; cy = self.ipw_y.ipw.center
        nz = self.ipw_z.ipw.normal; cz = self.ipw_z.ipw.center
        return three_plane_pt(nx, ny, nz, cx, cy, cz)
    #---------------------------------------------------------------------------
    # Traits callbacks
    #---------------------------------------------------------------------------

    @on_trait_change('show_cortex')
    def _show_cortex(self):
        if not self.anat_image:
            self.set(show_cortex=False, trait_change_notify=False)
            return
        elif not hasattr(self, 'cortical_surf'):
            self.add_cortical_surf()
        #self.toggle_cortex_visible(show.show_cortex)
        self.cortical_surf.visible = self.show_cortex

    @on_trait_change('show_anat')
    def _show_anat(self):
        if not self.anat_image:
            print 'no anat array loaded yet'
            self.set(show_anat=False, trait_change_notify=False)
            return
        # this will assess the status of the image source and plotting
        self.change_source_data()

    @on_trait_change('show_func')
    def _show_func(self):
        if not self.func_man or not self.func_man.overlay:
            print 'no functional manager to provide an overlay'
            self.set(show_func=False, trait_change_notify=False)
            return
        self.change_source_data()

    @on_trait_change('show_psurfs')
    def _show_pval_blobs(self):
        if not self.pscore_map:
            print 'no P score map is chosen'
            self.set(show_psurfs=False, trait_change_notify=False)
        #elif not hasattr(self, 'pval_surfs'):
        elif self.show_psurfs:
            self._update_p_map()
        self.pval_surfs.visible = self.show_psurfs

    @on_trait_change('func_man')
    def _link_stats(self):
        self.sync_trait('_pscores', self.func_man, alias='_stats_maps')
        self._set_blender_norm()

    @on_trait_change('alpha_compress')
    def _alpha_scale(self):
        if not self.func_man:
            return
        self.blender.over_alpha = self.func_man.alpha(scale=self.alpha_compress)

    @on_trait_change('func_man.norm')
    def _set_blender_norm(self):
        print 'resetting scalar normalization from func_man.norm'
        self.blender.over_norm = self.func_man.norm
        
    @on_trait_change('func_man.threshold')
    def _set_threshold(self):
        print 'remapping alpha because of func_man.threshold',
        if not self.func_man.overlay:
            print 'but no func_man'
            return
        print ''
        self.blender.over_alpha = self.func_man.alpha(scale=self.alpha_compress)

    @on_trait_change('func_man.cmap_option')
    def _set_over_cmap(self):
        self.blender.over_cmap = self.func_man.colormap
    
    @on_trait_change('func_man.overlay_updated')
    def _update_colors_from_func_man(self):
        """ When a new overlay is signaled, update the overlay color bytes
        """
        if not self.func_man or not self.func_man.overlay:
            return
        overlay = self.func_man.overlay
        # this could potentially change scalar mapping properties too
        self.blender.trait_setq(
            over_cmap=self.func_man.colormap,
            over_norm=self.func_man.norm,
            over_alpha=self.func_man.alpha(scale=self.alpha_compress)
            )
            
        self.blender.over = overlay.raw_image
                
    @on_trait_change('anat_image')
    def _update_colors_from_anat_image(self):
        """ When a new image is loaded, update the anat color bytes
        """
        self.__blocking_draw = True
        self.blender.main = self.anat_image.raw_image
        # hmmm there is a problem here when re-loading images..
        # the appropriate rgba array may not be available or valid yet
        self.change_source_data(new_position=True)
        self.__blocking_draw = False

        # flush previous arrays
        n_arr = self.anat_src.image_data.point_data.number_of_arrays
        names = [self.anat_src.image_data.point_data.get_array(i).name
                 for i in xrange(n_arr)]
        for n in names:
            self.anat_src.image_data.point_data.remove_array(n)
        # add new array
        image_arr = self.anat_image.image_arr.transpose().copy()
        self.anat_src.scalar_data = image_arr
        self.anat_src.spacing = self.blender.img_spacing
        self.anat_src.origin = self.blender.img_origin

    @on_trait_change('pscore_map')
    def _passthrough(self):
            self._update_p_map()
    
    @on_trait_change('func_man.overlay_updated')
    def _update_p_map(self):
        print 'updating p map'
        # if the functional overlay is updated, then also update the pvals
        if not (self.show_psurfs and self.pscore_map):
            print 'but no map chosen, or show surfs button unchecked'
            return
##         if arr is None:
        p_resamp = self.func_man.stats_overlay(self.pscore_map)
        if not p_resamp:
            print 'but no such map'
            return
        arr = p_resamp.image_arr.copy()
        if np.ma.getmask(arr) is not np.ma.nomask:
            arr = arr.filled()
        self.set_blob_src_scalars(arr)

    @on_trait_change('cluster_threshold')
    def _new_cluster_size(self):
##         arr = self.over_src.mlab_source.scalars
        self._update_p_map() #arr=arr)

    #---------------------------------------------------------------------------
    # Scene update methods
    #---------------------------------------------------------------------------
    def _stop_scene(self):
        self._render_mode = self.scene.disable_render
        self.scene.disable_render = True
    def _start_scene(self):
        mode = getattr(self, '_render_mode', True)
        self.scene.disable_render = mode

    def set_blob_src_scalars(self, arr):
        np.putmask(arr, arr <= 0.0, 1)
        components, nc = ndimage.label(arr <= P_THRESH)
        all_slices = ndimage.find_objects(components, nc)
        st = self.sig_threshold
        ct = self.cluster_threshold
        slices = filter(lambda s: ( arr[s] <= st ).sum() >= ct, all_slices)
        little_arrs = [arr[sl].copy() for sl in slices]
        arr[:] = 1.5*st
        for sl, l_arr in zip(slices, little_arrs):
            # make a filled in mask for this component
            msk = ndimage.binary_fill_holes(l_arr <= st)
            # get a scalar representation for this area --
            # using the masked mean
            sv = np.ma.masked_where(np.logical_not(msk), l_arr).mean()
            arr[sl] = np.where(msk, msk.astype(arr.dtype)*sv, arr[sl])
        self.blob_surf_src.mlab_source.set(scalars=arr)
        # ASSUMING THAT THE GRID SPACING AND ORIGIN IS EQUIVALENT IN
        # THE STATS MAPS AND THE BASE OVERLAY
        self.blob_surf_src.origin = [x for x,y in self.func_man.overlay.bbox]
        self.blob_surf_src.spacing = self.func_man.overlay.grid_spacing
        if not hasattr(self, 'pval_surfs'):
            self.add_significance_surfs()
        else:
            self._stop_scene()
            self.blob_surf_src.update_image_data = True
            self._start_scene()

    @on_trait_change('blender.main_rgba,blender.over_rgba,blender.blended_rgba')
    def _monitor_sources(self, obj, name, new):
        if self.__blocking_draw:
            return
        print name, 'changed'
        if name == 'main_rgba' and self.show_anat:
            self.change_source_data()
        elif name == 'over_rgba' and self.show_func:
            self.change_source_data()
    
    def change_source_data(self, new_position=False):
        """ Create a pixel-blended array, whose contents depends on the
        current plotting conditions. Also check the status of the
        visibility of the plots.

        """
        # XYZ: SAY FOR NOW THAT THE ANAT IMAGE MUST BE AVAILABLE BEFORE
        # PLOTTING THE OVERLAY... THIS WILL SIDESTEP THE ISSUE OF UPDATING
        # THE IMAGE DATA PROPERTIES, WHICH APPEARS TO CAUSE A HUGE DELAY
        


        if self.show_func and self.show_anat:
            print 'will plot blended'
            img_data = self.blender.blended_rgba
        elif self.show_func:
            print 'will plot over plot'
            img_data = self.blender.over_rgba
        else:
            print 'will plot anatomical'
            img_data = self.blender.main_rgba

        # if a new position, update (even if invisibly)
        if new_position:
            # this will kick off the scalar_data_changed stuff
            self.blended_src.scalar_data = img_data.copy()
            self.blended_src.spacing = self.blender.img_spacing
            self.blended_src.origin = self.blender.img_origin
            self.blended_src.update_image_data = True #???
        
        if not self.show_func and not self.show_anat:
            self.toggle_planes_visible(False)
            return

        self._stop_scene()

        if not new_position:
            print 'changing data in-place'
            self.blended_src.scalar_data[:] = img_data

        #self.blended_src.update_image_data = True
        self.blended_src.update()
##         cProfile.runctx('self.blended_src.update()', globals(), locals(),
##                         'mayavi.prof')
##         s = pstats.Stats('mayavi.prof')
##         s.strip_dirs().sort_stats('cumulative').print_stats()

        if not hasattr(self, 'ipw_x'):
            t0 = time.time()
            print 'also adding plots to scene ',
            self.add_plots_to_scene()
            t = time.time()
            print 'done, %1.3f sec'%(t-t0)
        else:
            t0 = time.time()
            print 'also turning on plots ',
            self.toggle_planes_visible(True)
            t = time.time()
            print 'done, %1.3f sec'%(t-t0)

        t0 = time.time()
        print 'also starting scene ',
        self._start_scene()
        # this seems hacky, but works
        self.scene.render_window.render()
        t = time.time()
        print 'done, %1.3f sec'%(t-t0)

    def toggle_planes_visible(self, value):
        self._toggle_poly_extractor_mode(cut_mode=value)
        for ax in ('x', 'y', 'z'):
            ipw = getattr(self, 'ipw_%s'%ax, False)
            if ipw:
                ipw.visible = value

    def _toggle_poly_extractor_mode(self, cut_mode=True):
        if not self.poly_extractor.implicit_function:
            self.poly_extractor.implicit_function = tvtk.Planes()
        pfunc = self.poly_extractor.implicit_function
        if not hasattr(self, 'ipw_x'):
            cut_mode = False
        if cut_mode:
            pts = [ [] ] * 3
            normals = [ [] ] * 3
            for ax, index in self._axis_index.iteritems():
                ipw = getattr(self, 'ipw_%s'%ax).ipw
                pts[index] = ipw.center
                normals[index] = map(lambda x: -x, ipw.normal)
            pfunc.points = pts
            pfunc.normals = normals
        else:
            # set up the poly extractor filter with an all-inclusive
            # Implicit Function
            scene_points = [0]*24
            try:
                self.scene.camera.get_frustum_planes(1, scene_points)
                pfunc.set_frustum_planes(scene_points)
            except:
                # guess scene isn't set up yet
                pass
        
        self.poly_extractor.extract_inside = not cut_mode
            
            
    #---------------------------------------------------------------------------
    # Scene activation callbacks 
    #---------------------------------------------------------------------------
    @on_trait_change('scene.activated')
    def display_scene3d(self):
        print 'making 3d scene'
        self.scene.mlab.view(100, 100)
        self.scene.scene.background = (0, 0, 0)
        # set up the poly extractor filter with an all-inclusive
        # Implicit Function
        self._toggle_poly_extractor_mode(cut_mode=False)
        # Keep the view always pointing up
        self.scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
##         self.scene.picker.pointpicker.add_observer('EndPickEvent',
##                                                    self.pick_callback)
##         self.scene.picker.show_gui = False

    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(
        VGroup(
            HGroup(
                Item('scene',
                     editor=SceneEditor(scene_class=MayaviScene),
                     height=450, width=300),
                show_labels=False,
                dock='vertical'
                ),
            HGroup(
                Item('show_anat', label='Show anatomical'),
                Item('show_func', label='Show functional'),
                Item('show_cortex', label='Show cortex'),
                Item('alpha_compress',style='custom',label='Alpha compression')
                ),
            HGroup(
                Item('pscore_map', label='P Score Map'),
                Item('show_psurfs', label='Show Significance Blobs'),
                Item('cluster_threshold', label='Min Cluster Size'),
                Item('sig_threshold', label='Significance Level',
                     style='custom')
                )
            ),
        resizable=True,
        title='MEG Activations',
        )


class MayaviWidget(TopLevelAuxiliaryWindow):

    def __init__(self, parent=None, main_ref=None, functional_manager=None,
                 **traits):
        TopLevelAuxiliaryWindow.__init__(self,
                                         parent=parent,
                                         main_ref=main_ref)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        traits['func_man'] = functional_manager
        self.mr_vis = OrthoView3D(**traits)
        layout.addWidget(self.mr_vis.edit_traits(parent=self,
                                                 kind='subpanel').control)
        self.func_widget = None
        self.layout_box = layout
        if functional_manager is not None:
            self.add_toolbar(functional_manager)
##             self.func_widget = overlay_panel_factory(functional_manager,
##                                                      parent=self)
##             layout.addWidget(self.func_widget)
##             print 'trying to add the functional manager UI'
##             layout.addWidget(
##                 functional_manager.edit_traits(
##                     parent=self,
##                     kind='subpanel'
##                     ).control
##                 )
        else:
            print 'no functional manager'
        self.setObjectName('3D Plot')

    def add_toolbar(self, functional_manager):
        self.mr_vis.func_man = functional_manager
        if self.func_widget is not None:
            print 'removing old func widget'
            self.layout_box.removeWidget(self.func_widget)
            self.func_widget.close()
        else:
            print 'not removing old func widget'
        self.func_widget = functional_manager.make_panel(parent=self)
        self.layout_box.addWidget(self.func_widget)
        self.update()

if __name__=='__main__':
    from PyQt4 import QtCore, QtGui
    import sys
    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance() 

    win = MayaviWidget()
    
    win.show()
    app.exec_()
                        
    

    
