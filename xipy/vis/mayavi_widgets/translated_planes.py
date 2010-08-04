# NumPy
import numpy as np

# NIPY
from nipy.core import api as ni_api

# Enthought library
import enthought.traits.api as t
import enthought.traits.ui.api as tui

from enthought.tvtk.api import tvtk
from enthought.mayavi import mlab

from enthought.mayavi.filters.set_active_attribute import SetActiveAttribute

# XIPY imports
from xipy.colors.mayavi_tools import image_plane_widget_rgba
from xipy.vis.mayavi_widgets import VisualComponent

axis_to_index = dict( zip('xyz', [0,1,2]) )

class TranslatedPlanes(VisualComponent):

    # Control the planes to mirror
    mx = t.Bool(False)
    my = t.Bool(False)
    mz = t.Bool(False)
    ma = t.Bool(False)

    # Control the offset of the fixed planes along its axis
    offset = t.Range(low=10., high=50., value=20.)
    edges = [0, 0, 0]

##     # Control the channels to plot
##     _available_surfaces = t.Property(depends_on='display.blender.over')
##     surface_component = t.Enum(values='_available_surfaces')

    # active_attribute filter to sit on top of the master source
    aa = SetActiveAttribute()
    color_chan = t.Str
    # Volume box
    bbox = t.Property

    # View
    view = tui.View(
        tui.HGroup(
            tui.VGroup(
                tui.HGroup(
                    tui.Item('mx', label='Mirror X'),
                    tui.Item('my', label='Mirror Y')
                    ),
                tui.HGroup(
                    tui.Item('mz', label='Mirror Z'),
                    tui.Item('ma', label='Mirror all')
                    )
                ),
            tui.VGroup(
                tui.Item('color_chan',
                         editor=tui.EnumEditor(
                             name='object.aa._point_scalars_list'
                             ),
                         #style='custom',
                         label='Mirrored Channel'),
                tui.Item('offset', label='Plane offsets')
                )
            ),
        )

    # ------------------------------------------------------------------------
    def __init__(self, display, **traits):
        if 'name' not in traits:
            traits['name'] = 'Fixed Planes'
        traits['display'] = display
        VisualComponent.__init__(self, **traits)
        for trait in ('master_src',):
            self.add_trait(trait, t.DelegatesTo('display'))
        
        self.aa = mlab.pipeline.set_active_attribute(self.master_src)
        self.sync_trait('color_chan', self.aa, alias='point_scalars_name')
        #self._point_scalars_list = t.DelegatesTo('aa')

##     # XXX: commmon usage in cortical surface component
##     @t.cached_property
##     def _get__available_surfaces(self):
##         return [component_to_surf[ch] for ch in self.master_src.rgba_channels]

    def _get_bbox(self):
        # XXX: TRANSLATING HACK!
        box_ext = np.array(self.master_src.blender.image_arr.shape[:3])
        box_ext *= np.abs(self.master_src.blender.coordmap.affine.diagonal()[:3])
        return zip( (0, 0, 0), box_ext )
##         return self.master_src.blender.bbox

    def __all_set(self, state, quiet=True):
        self.set(trait_change_notify=not quiet,
                 mx=state, my=state, mz=state, ma=state)

    @t.on_trait_change('mx, my, mz, ma')
    def toggle_planes(self, obj, name, value):
        # hacky
        if name=='ma':
            axes = 'xyz'
            state = getattr(self, name)
            print 'setting all states to', state
            self.__all_set(state)
##             self.trait_setq(mx=state, my=state, mz=state)
            bools = (self.mx, self.my, self.mz)
        else:
            axes = name[1]
            bools = ( getattr(self, name), )
        for ax, state in zip(axes, bools):
            if state:
                print 'adding plane to pipeline'
                try:
                    self.add_fixed_plane(ax)
                except RuntimeError:
                    self.__all_set(False)
                    return
            else:
                print 'removing resliced image from pipeline'
                r_img = getattr(self, 'resliced_img_%s'%ax)
                r_img.remove()
            self.setup_listener(ax, state)

    @t.on_trait_change('offset')
    def set_offsets(self):
        for ax in 'xyz':
            ipw = getattr(self, 'm_ipw_%s'%ax, None)
            if not ipw:
                continue
            rfilter = getattr(self, 'resliced_img_%s'%ax)
            # 1st, set the new position
            ax_idx = axis_to_index[ax]
            ipw.ipw.slice_position = self.bbox[ax_idx][0] - self.offset
            # 2nd, set the translation vector
            self.set_translation(rfilter.filter, ax)

    def setup_listener(self, axis, state):
        main_ipw = self.display._ipw_x(axis)
        if state:
            cb = getattr(self, '_%s_pos_listener'%axis)
            n1 = main_ipw.ipw.add_observer('InteractionEvent', cb)
            n2 = main_ipw.ipw.add_observer('EndInteractionEvent', cb)
            setattr(self, '_%s_listener_ids'%axis, (n1, n2))
        else:
            ids = getattr(self, '_%s_listener_ids'%axis)
            for n in ids:
                main_ipw.ipw.remove_observer(n)
    # -- Listener passthrough methods ----------------------------------------
    def _x_pos_listener(self, widget, event):
        filter = self.resliced_img_x.filter
        self.set_translation(filter, 'x')
    def _y_pos_listener(self, widget, event):
        filter = self.resliced_img_y.filter
        self.set_translation(filter, 'y')
    def _z_pos_listener(self, widget, event):
        filter = self.resliced_img_z.filter
        self.set_translation(filter, 'z')
    # -- Sets translation vector in the reslice affine -----------------------
    def set_translation(self, filter, axis):
        main_ipw = self.display._ipw_x(axis)
        axis_idx = axis_to_index[axis]
        main_ipw.ipw.update_traits()
        pos = main_ipw.ipw.slice_position
        new_pos = [0] * 3
        new_pos[axis_idx] = pos-(self.bbox[axis_idx][0] - self.offset)
        filter.reslice_axes_origin = new_pos
    # -- Adds a new fixed plane along "axis" ---------------------------------
    def add_fixed_plane(self, axis):
        "Add an offset plane for current channel on `axis` in {'x', 'y', 'z'}"
        if self.display._ipw_x(axis) is None:
            raise RuntimeError('No Axis to mirror!')
        if not self.aa.point_scalars_name:
            raise RuntimeError('No Colors to plot!')
        axis_idx = axis_to_index[axis]
        # set up the new fixed image plane widget at position w0
        w0 = self.bbox[axis_idx][0] - self.offset
        r = tvtk.ImageReslice()
        self.set_translation(r, axis)
        resliced_img = mlab.pipeline.user_defined(self.aa, filter=r)
        m_ipw = image_plane_widget_rgba(resliced_img)
        m_ipw.ipw.interaction = 0
        m_ipw.ipw.plane_orientation = '%s_axes'%axis
        m_ipw.ipw.restrict_plane_to_volume = False
        m_ipw.ipw.slice_position = w0
        setattr(self, 'm_ipw_%s'%axis, m_ipw)
        setattr(self, 'resliced_img_%s'%axis, resliced_img)
