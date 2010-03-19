import os
os.environ['ETS_TOOLKIT'] = 'qt4'

from PyQt4 import QtGui, QtCore

from enthought.traits.api import HasTraits, Instance, on_trait_change, Array, \
     Bool, Range, Enum, Property, List
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
from matplotlib import cm

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

P_THRESH = 0.05
CLUSTER_THRESH = 3

class OrthoView3D(HasTraits):
    #---------------------------------------------------------------------------
    # Data and Figure
    #---------------------------------------------------------------------------
##     anat_data = Array()
##     over_data = Array()
##     pstat_data = Array()
    scene = Instance(MlabSceneModel, ())

    anat_src = Instance(Source)
##     over_src = Instance(Source)
    blended_src = Instance(Source)
    _blended_src_scalars = Array(dtype='B')
    blob_surf_src = Instance(Source)

##     anat_image = Instance(VolumeSlicerInterface)
    # BE RESTRICTIVE ABOUT THE TYPE OF IMAGE COMING IN, UNTIL I FIGURE
    # OUT RESAMPLING IN VTK
    anat_image = Instance(ResampledVolumeSlicer)
    # the anatomical image, normalized to [0,255] and mapped
    # into RGBA components
    _anat_rgba_bytes = Array(dtype='B')

    #---------------------------------------------------------------------------
    # Functional Overlay Manager
    #---------------------------------------------------------------------------
    func_man = Instance(OverlayInterface)
    # the overlay image, normalized to [0,255] and mapped
    # into RGBA components
    _over_rgba_bytes = Array(dtype='B')

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
    alpha_scaling = Range(low=0.0, high=4.0, value=1.0,
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
        

    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _anat_src_default(self):
        s = ArraySource()
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
            data_shape = locals()['data_shape']
            ipw.ipw.slice_position = data_shape[dim]/4
            # name this widget and attribute it to self
            setattr(self, 'ipw_%s'%ax, ipw)
            ipw.ipw.add_observer(
                'InteractionEvent',
                getattr(self, '_%s_planes_interaction'%ax)
                )
            ipw.ipw.add_observer(
                'EndInteractionEvent',
                self._update_for_cortex
                )
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
            point_scalars=surf_name
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
    def _x_planes_interaction(self, widget, event):
        self._link_plane_point('x')
    def _y_planes_interaction(self, widget, event):
        self._link_plane_point('y')
    def _z_planes_interaction(self, widget, event):
        self._link_plane_point('z')
    def _link_plane_point(self, ax):
        ax_idx = self._axis_index[ax]
        ipw = getattr(self, 'ipw_%s'%ax).ipw
        planes_function = self.poly_extractor.implicit_function
        planes_function.points[ax_idx] = ipw.center
        planes_function.normals[ax_idx] = map(lambda x: -x, ipw.normal)
    def _update_for_cortex(self, widget, event):
        if hasattr(self, 'cortical_surf') and self.show_cortex:
            self.anat_src.update()
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
        self.blend_sources()
##         elif not hasattr(self, 'main_ipw_x'):
##             self._update_main_from_anat_image()
##         self.toggle_main_visible(self.show_anat)

    @on_trait_change('show_func')
    def _show_func(self):
        if not self.func_man:
            print 'no functional manager to provide an overlay'
            self.set(show_func=False, trait_change_notify=False)
        self.blend_sources()
##         elif not hasattr(self, 'over_ipw_x'):            
##             self._update_overlay_from_func_man()
##         self.toggle_over_visible(self.show_func)

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

    @on_trait_change('alpha_scaling')
    def _alpha_scale(self):
        self._update_colors_from_func_man()

    @on_trait_change('func_man.threshold')
    def _set_threshold(self):
        # now the func_man.alpha(threshold=True) will reflect the thresholding
        self._update_colors_from_func_man()
        # quickly remap the alpha component of over_rgba_bytes, this will
        # be appx 4x faster than mapping all components
        # XYZ: do this later
##         alpha_chan = self._over_rgba_bytes[...,3]
##         alpha_chan[:] = alpha.take(self._over_lut_idx, mode='clip')
    
    @on_trait_change('func_man.overlay_updated')
    def _update_colors_from_func_man(self):
        """ When a new overlay is signaled, update the overlay color bytes
        """
        if not self.func_man or not self.func_man.overlay:
            return
        overlay = self.func_man.overlay
        assert type(overlay) is ResampledVolumeSlicer, 'Mayavi widget can only handle ResampledVolumeSlicer image types'
        arr = overlay.image_arr.transpose()
        self._over_rgba_bytes = rgba_blending.normalize_and_map(
            arr, cm.jet,
            alpha=self.func_man.alpha(scale=self.alpha_scaling),
            norm_min=self.func_man.norm[0], norm_max=self.func_man.norm[1]
            )
        self.blend_sources()
                
    @on_trait_change('anat_image')
    def _update_colors_from_anat_image(self):
        """ When a new image is loaded, update the anat color bytes
        """
        arr = self.anat_image.image_arr
        # map with grayscale, alpha=1 (except for first few points)
        a = np.ones(256, 'B')*255
        a[:5] = 0
        self._anat_rgba_bytes = rgba_blending.normalize_and_map(
            arr.transpose(), cm.gray, alpha=a
            )
        
        self._blended_src_scalars = self._anat_rgba_bytes.copy()
        # this will kick off the scalar_data_changed stuff
        self.blended_src.scalar_data = self._blended_src_scalars
        self.blended_src.spacing = self.anat_image.grid_spacing
        self.blended_src.origin = np.array(self.anat_image.bbox)[:,0]
        self.blended_src.update_image_data = True #???
        self.blend_sources()

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

    def blend_sources(self):
        """ Create a pixel-blended array, whose contents depends on the
        current plotting conditions. Also check the status of the
        visibility of the plots.

        """
        # XYZ: SAY FOR NOW THAT THE ANAT IMAGE MUST BE AVAILABLE BEFORE
        # PLOTTING THE OVERLAY... THIS WILL SIDESTEP THE ISSUE OF UPDATING
        # THE IMAGE DATA PROPERTIES, WHICH APPEARS TO CAUSE A HUGE DELAY
        
        if not self.show_func and not self.show_anat:
            self.toggle_planes_visible(False)
            return
        self._stop_scene()

        # many combinations here..
        # plot anatomical only
        # plot anatomical with functional blended
        # plot functional only (needs to blend into a zero-alpha anatomical)
        
        # the bytes arrays have been transposed, so reflect this
        # in the grid orientation specs
        main_dr = self.anat_image.grid_spacing[::-1]
        main_r0 = np.array(self.anat_image.bbox)[::-1,0]
        t0 = time.time()
        print 'setting scalar data to anat bytes ',
        self._blended_src_scalars[:] = self._anat_rgba_bytes
        t = time.time()
        print 'done, %1.3f sec'%(t-t0)
        if self.show_func:
            t0 = time.time()
            print 'blending in functional bytes ',
            if not self.show_anat:
                self._blended_src_scalars[...,3] = 0
            over_dr = self.func_man.overlay.grid_spacing[::-1]
            over_r0 = np.array(self.func_man.overlay.bbox)[::-1,0]
            over_bytes = self._over_rgba_bytes
            blended = rgba_blending.resample_and_blend(
                self._blended_src_scalars, main_dr, main_r0,
                over_bytes, over_dr, over_r0
                )
            t = time.time()
            print 'done, %1.3f sec'%(t-t0)
        
        #self.blended_src.update_image_data = True

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
                Item('alpha_scaling', style='custom', label='Alpha scaling')
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

    def __init__(self, functional_manager=None,
                 parent=None, main_ref=None, **traits):
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
                        
    

    
