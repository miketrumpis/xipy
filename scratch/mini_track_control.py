import xipy
import get_tracks
import plane_intersection
import xipy_fos
import enthought.traits.api as t_api
import enthought.traits.ui.api as tui_api
from enthought.mayavi import mlab
from enthought.tvtk.api import tvtk

import numpy as np

class mini_track_feature(t_api.HasTraits):

    #mayavi_win = t_api.Instance('xipy.vis.mayavi_widgets.OrthoView3D')
    b = t_api.Button('Plot Lines')
    x_lines = t_api.Bool(False)
    #plane = t_api.Enum(values='mayavi_win._axis_index.keys()')
    
    def __init__(self, tracks, mwin, reduction=10, **traits):
        t_api.HasTraits.__init__(self, **traits)
        if type(tracks) is str:
            self.tracks = get_tracks.load_reduce_translate(
                tracks, reduction=reduction
                )
        elif type(tracks) is list:
            self.tracks = tracks
        else:
            raise ValueError("don't know what to do with tracks!")
        self.colors = get_tracks.simple_directional_colors(self.tracks)
        self.track_lines = None
        self.mayavi_win = mwin
##         self.add_trait('plane', t_api.Enum(*mwin._axis_index.keys()))
        self.on_trait_change(self.add_lines_to_scene, 'b', dispatch='new')

##     def _b_fired(self):
##         if self.mayavi_win is None:
##             return
##         self.add_lines_to_scene()        

    def _get_intersection(self):
        if self.mayavi_win is None:
            return
        ipw = getattr(self.mayavi_win, 'ipw_%s'%self.plane).ipw
        p0 = ipw.origin
        nm = ipw.normal
        print 'finding intersections in plane:', p0, nm
        x_pts = plane_intersection.intersecting_tracks(self.tracks, p0, nm)
        return x_pts

    def add_lines_to_scene(self):
        if self.mayavi_win is None:
            return
        if self.x_lines:
            print 'find intersection'
            x_pts = self._get_intersection()
            tracks = [self.tracks[i] for i in x_pts]
##             colors = np.hstack( (self.colors[x_pts], np.ones((len(x_pts), 1)) ))
            colors = self.colors[x_pts]
        else:
            tracks = self.tracks
##             colors = np.hstack((self.colors, np.ones((len(self.colors), 1)) ))
            colors = self.colors

        if self.track_lines:
            self.track_lines.remove()
        track_lines, lut = xipy_fos.tvtk_line(tracks, colors, opacity=.5)
##         colors = (colors*255).astype('B')
##         print colors
## ##         colors = np.ones_like(colors)
##         print colors.shape, len(tracks)
##         rgba_colors = tvtk.UnsignedCharArray()
##         rgba_colors.from_array(colors)
##         print rgba_colors
##         track_lines.point_data.scalars = rgba_colors
##         track_lines.point_data.scalars.name = 'lines'
        self.track_lines = mlab.pipeline.add_dataset(track_lines, figure=self.mayavi_win.fig)
        self.lsurf = mlab.pipeline.surface(self.track_lines, figure=self.mayavi_win.fig)
        self.lsurf.module_manager.scalar_lut_manager.lut_mode = 'file'
        self.lsurf.module_manager.scalar_lut_manager.lut = lut
        self.lsurf.actor.set_lut(lut)
        self.lsurf.actor.property.line_width = 1
        self.lsurf.actor.property.opacity = .5
##         self.lsurf.render()
##         if self.track_lines:
##             self.mayavi_win.scene.remove_actor(self.track_lines)
##         self.mayavi_win.scene.add_actor(track_lines)
        

    view = tui_api.View(
        tui_api.HGroup(
            tui_api.Item('b', show_label=False),
##             tui_api.Item('x_lines', label='Only Show Intersecting Lines'),
##             tui_api.Item('plane', label='Intersection Plane')
            ),
        resizable=True
        )


if __name__=='__main__':
    from enthought.tvtk.pyface import picker
    from enthought.mayavi.sources.array_source import ArraySource

    class mayavi_win(object):
        def __init__(self):
            self.fig = mlab.figure()
            self.scene = self.fig.scene
            src = ArraySource(transpose_input_array=False)
            src.scalar_data = np.random.randn(10,10,10)
            src.origin = [-50,-50,-50]
            src.spacing = [10,10,10]
            mlab.pipeline.image_plane_widget(src, figure=self.fig)
        
    class pick_handler(picker.PickHandler):
        def __init__(self, mayavi_window, track_plotter, **traits):
            picker.PickHandler.__init__(self, **traits)
            self.mwin = mayavi_window
            self.tman = track_plotter
            self.picked_lines = []
        
        def handle_pick(self, data):
            self.data = data
            if data.cell_id < 0:
                self.restore_tracks()
            else:
                self.highlight_track(data.cell_id)

        def restore_tracks(self):
            while self.picked_lines:
                line = self.picked_lines.pop()
                line.remove()
            self.tman.lsurf.actor.property.opacity = .5

        def highlight_track(self, cell_id):
            p = self.mwin.scene.camera.position
            trk = self.tman.tracks[cell_id]
            rgb = self.tman.colors[cell_id]
            pd, lut = xipy_fos.tvtk_line([trk], [rgb])
            pline = mlab.pipeline.add_dataset(pd, figure=self.mwin.fig)
            self.picked_lines.append( pline )
            lsurf = mlab.pipeline.surface(pline, figure=self.mwin.fig)
            lsurf.module_manager.scalar_lut_manager.lut_mode = 'file'
            lsurf.module_manager.scalar_lut_manager.lut = lut
            lsurf.actor.set_lut(lut)
            lsurf.actor.property.line_width = 2
            lsurf.actor.property.opacity = 1
            self.tman.lsurf.actor.property.opacity = .01
            #self.mwin.scene.camera.position = p
            lsurf.render()
            self.mwin.scene.render()
            
    mw = mayavi_win()
    my_track_file = '/Users/mike/workywork/dipy-vis/brain1/brain1_scan1_fiber_track_mni.trk'
    mf = mini_track_feature(my_track_file, mw, reduction=100)

    ph = pick_handler(mw, mf)

    mw.scene.picker.pick_handler = ph
    mw.scene.picker.pick_type = 'cell_picker'

    mf.edit_traits()
##     mlab.show()
