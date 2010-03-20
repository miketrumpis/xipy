import get_tracks
import plane_intersection
import xipy_fos
import enthought.traits.api as t_api
import enthought.traits.ui.api as tui_api

class mini_track_feature(t_api.HasTraits):

    #mayavi_win = t_api.Instance('xipy.vis.mayavi_widgets.OrthoView3D')
    b = t_api.Button('Plot Lines')
    x_lines = t_api.Bool(False)
    #plane = t_api.Enum(values='mayavi_win._axis_index.keys()')
    
    def __init__(self, tracks, mwin, **traits):
        t_api.HasTraits.__init__(self, **traits)
        if type(tracks) is str:
            self.tracks = get_tracks.load_reduce_translate(
                tracks, reduction=10
                )
        elif type(tracks) is list:
            self.tracks = tracks
        else:
            raise ValueError("don't know what to do with tracks!")
        self.colors = get_tracks.simple_directional_colors(self.tracks)
        self.track_lines = None
        self.mayavi_win = mwin
        self.add_trait('plane', t_api.Enum(*mwin._axis_index.keys()))
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
            colors = self.colors[x_pts]
        else:
            tracks = self.tracks
            colors = self.colors
        self.mayavi_win._stop_scene()
            
        track_lines = xipy_fos.tvtk_line(tracks, colors, opacity=.25)
        if self.track_lines:
            self.mayavi_win.scene.remove_actor(self.track_lines)
        self.mayavi_win.scene.add_actor(track_lines)
        self.mayavi_win._start_scene()
        self.track_lines = track_lines

    view = tui_api.View(
        tui_api.HGroup(
            tui_api.Item('b', show_label=False),
            tui_api.Item('x_lines', label='Only Show Intersecting Lines'),
            tui_api.Item('plane', label='Intersection Plane')
            ),
        resizable=True
        )
