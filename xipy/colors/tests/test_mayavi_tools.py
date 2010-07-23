import nose.tools as nt
import numpy as np
import nipy.core.api as ni_api

from xipy.slicing import xipy_ras

from xipy.colors.mayavi_tools import *
from xipy.colors.rgba_blending import quick_convert_rgba_to_vtk_array
from enthought.mayavi import mlab


main_img = ni_api.Image(
    np.ones((30,40,20)),
    ni_api.AffineTransform.from_start_step('ijk', xipy_ras, [0,0,0], [1,1,1])
    )

over_img = ni_api.Image(
    np.ones((10,18,20)),
    ni_api.AffineTransform.from_start_step('ijk', xipy_ras, [0,0,0], [3,2,1])
    )

def test_master_source_with_extra_channels():
    bi = BlendedImages(vtk_order=True)
    m = MasterSource(blender=bi)
    aa = mlab.pipeline.set_active_attribute(m)
    bi.main = main_img

    m.set_new_array(np.ones(np.prod(main_img.shape)), 'extra')

    yield nt.assert_true, 'extra' in aa._point_scalars_list, \
          'New channel not available'

    aa.point_scalars_name = 'extra'
    ipw = mlab.pipeline.image_plane_widget(aa)

    bi.over = over_img

    yield nt.assert_true, 'extra' in aa._point_scalars_list, \
          'New channel not available'
    yield nt.assert_true, m.over_channel in aa._point_scalars_list

    m.safe_remove_arrays(names=['extra'])
    

def test_master_source_states():
    "Tests that the MasterSource states are consistent with the data it follows"
    bi = BlendedImages(vtk_order=True)
    m = MasterSource(blender=bi)
    aa = mlab.pipeline.set_active_attribute(m)
    bi.over = over_img

    # now,
    # * the scalar data/name should reflect bi.over_rgba
    # * the channel names should be MasterSource.over_channel
    # * aa._point_scalars_list should have MasterSource.over_channel
    scalar_data = m.data.point_data.scalars.to_array()
    yield (
        nt.assert_true,
        (scalar_data == quick_convert_rgba_to_vtk_array(bi.over_rgba)).all(),
        'Data inconsistent'
        )
    yield nt.assert_true, m.over_channel in m.rgba_channels, \
          'Channel name not available'
    yield nt.assert_true, m.data.point_data.scalars.name == m.over_channel, \
          'Scalar name inconsistent'
    yield nt.assert_true, m.over_channel in aa._point_scalars_list, \
          'Channel name not available in downstream AA'
    del scalar_data


    bi.main = main_img

    # now,
    # * the scalar data/name should reflect bi.main_rgba
    # * the channel names should be full of (over, main, blended)
    # * aa._point_scalars_list should have all of m.rgba_channels
    scalar_data = m.data.point_data.scalars.to_array()
    yield (
        nt.assert_true,
        (scalar_data == quick_convert_rgba_to_vtk_array(bi.main_rgba)).all(),
        'Data inconsistent'
        )
    yield nt.assert_true, len(m.rgba_channels)==3, \
          'Channel names not all available'
    yield nt.assert_true, m.data.point_data.scalars.name == m.main_channel, \
          'Scalar name inconsistent'
    yield (
        nt.assert_true,
        all([name in aa._point_scalars_list for name in m.rgba_channels]), 
        'Channel name not available in downstream AA'
        )
    del scalar_data

    bi.over = None
    # now,
    # * the channel names should just have bi.main_rgba
    # * the downstream AA should reflect this too
    scalar_data = m.data.point_data.scalars.to_array()
    yield (
        nt.assert_true,
        (scalar_data == quick_convert_rgba_to_vtk_array(bi.main_rgba)).all(),
        'Data inconsistent'
        )
    yield (
        nt.assert_true,
        len(m.rgba_channels)==1 and m.main_channel in m.rgba_channels,
        'Channel names inconsistent'
        )
    yield nt.assert_true, m.data.point_data.scalars.name == m.main_channel, \
          'Scalar name inconsistent'
    yield (
        nt.assert_true,
        len(aa._point_scalars_list) == 2 and m.main_channel in aa._point_scalars_list,
        'Channel name not available in downstream AA'
        )
