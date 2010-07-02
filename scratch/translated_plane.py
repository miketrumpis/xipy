import numpy as np
import xipy.vis.rgba_blending as rb
import xipy
import xipy.io as xio
import xipy.vis.mayavi_tools as mt
import xipy.vis.mayavi_widgets as mw

from enthought.mayavi import mlab
from enthought.tvtk.api import tvtk

over_img = xio.load_spatial_image('tfbeam_img.nii')
over_img._data = np.ma.masked_where(over_img._data > 1e10, over_img._data)

main_img = xio.load_spatial_image(xipy.TEMPLATE_MRI_PATH)

bi = rb.BlendedImages(vtk_order = True)
m_src = mw.MasterSource(blender=bi)
bi.main = main_img
bi.over = over_img

m_src = mlab.pipeline.add_dataset(m_src)

aa1 = mlab.pipeline.set_active_attribute(
    m_src, point_scalars=m_src.main_channel
    )

ipw1 = mt.image_plane_widget_rgba(aa1)
ipw1.ipw.plane_orientation = 'x_axes'

aa2 = mlab.pipeline.set_active_attribute(
    m_src, point_scalars=m_src.over_channel
    )

bbox = bi.bbox
x_extent = bbox[0][1] - bbox[0][0]
# want to translate the image along x-axis for x_extent + 20 mm

reslice = tvtk.ImageReslice()
resliced_img = mlab.pipeline.user_defined(aa2, filter=reslice)

ipw2 = mt.image_plane_widget_rgba(resliced_img)
x0 = bbox[0][0] - 10
p2_pos = x0
## ipw2.ipw.origin = p2_pos
ipw2.ipw.slice_position = p2_pos
ipw2.ipw.interaction = 0

#ipw2.ipw.origin = p2_pos

def update_offset(widget, event):
    ipw = tvtk.to_tvtk(widget)
    translate = reslice.reslice_axes_origin
    translate[0] = ipw.slice_position - x0
    print 'translation:', translate
    reslice.reslice_axes_origin = ipw.slice_position-x0-10, 0, 0

## ipw1.ipw.add_observer('InteractionEvent', update_offset)
ipw1.ipw.add_observer('EndInteractionEvent', update_offset)


