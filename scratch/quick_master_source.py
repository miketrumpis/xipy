import numpy as np
import xipy.colors.rgba_blending as rb
import xipy
import xipy.utils as utils
import xipy.io as xio
import xipy.colors.mayavi_tools as mt
import xipy.vis.mayavi_widgets as mw

import enthought.traits.api as traits
import enthought.traits.ui.api as tui
from enthought.mayavi import mlab
from enthought.tvtk.api import tvtk

class SimpleSwitching(traits.HasTraits):

    source = traits.Instance(mw.MasterSource, ())

    add_new_channel = traits.Button('add chan')

    new_overlay = traits.Button('switch overlay')

    plot = traits.Button('Plot on AA')

    aa = traits.Instance('enthought.mayavi.filters.filter_base.Filter')

    def __init__(self, **t):
        traits.HasTraits.__init__(self, **t)
        self.n_added = 0
        self._current_img = over_img
        self.aa = mlab.pipeline.set_active_attribute(self.source)
        self.on_trait_change(self.flush_aa, 'source.pipeline_changed')
##         self.source.blender.over = self._current_img

    def flush_aa(self):
        self.aa.update_pipeline()

    @traits.on_trait_change('add_new_channel')
    def add_chan(self):
        shape = self.source.blender.main_rgba.shape[:3]
        img = np.zeros(shape, dtype='B')
        center = np.array(shape)/2.
        vx = utils.voxel_index_list(shape) - center
        sz = np.random.randint(4, high=15)**2
        good_vx = np.where( (vx**2).sum(axis=1) < sz )[0]
        np.put(img, good_vx, 1)
        self.source.set_new_array(img, 'new_img_%d'%self.n_added, update=True)
        self.n_added += 1

    @traits.on_trait_change('new_overlay')
    def swap_images(self):
        blender = self.source.blender
        if self._current_img is over_img2:
            self._current_img = over_img
        else:
            self._current_img = over_img2
        blender.over = self._current_img

    def _plot_fired(self):
        if not self.aa.point_scalars_name:
            print 'no point scalars to plot'
            return
        mt.image_plane_widget_rgba(self.aa)
    
    view = tui.View(
        tui.VGroup(
            tui.Group(
                tui.Item('add_new_channel', show_label=False),
                tui.Item('new_overlay', show_label=False),
                tui.Item('plot', show_label=False)
                ),
            tui.Group(
                tui.Item('object.source.rgba_channels',
                         label='Source RGBA Channels'), #style='readonly'),
                tui.Item('object.source.all_channels',
                         label='Source Channels') #, style='readonly')
                ),
            tui.Group(
                tui.Item('object.aa', style='custom')
                )
            ),
        resizable=True
        )

if __name__=='__main__':
    over_img = xio.load_spatial_image('tfbeam_img.nii')
    over_img._data = np.ma.masked_where(over_img._data > 1e10, over_img._data)

    over_img2 = xio.load_spatial_image('map_img.nii')

    main_img = xio.load_spatial_image(xipy.TEMPLATE_MRI_PATH)

    bi = rb.BlendedImages(vtk_order = True, over_alpha = .5)
    m_src = mw.MasterSource(blender=bi)
    bi.main = main_img
##     bi.over = over_img
    m_src = mlab.pipeline.add_dataset(m_src)
    switcher = SimpleSwitching(source=m_src)
    ui = switcher.edit_traits()
    aa1 = mlab.pipeline.set_active_attribute(
        m_src, point_scalars=m_src.main_channel
        )
    ipw1 = mt.image_plane_widget_rgba(aa1)
    ipw1.ipw.plane_orientation = 'x_axes'

##     aa2 = mlab.pipeline.set_active_attribute(
##         m_src, point_scalars=m_src.over_channel
##     )

##     ipw2 = mt.image_plane_widget_rgba(aa2)
    
    mlab.show()
