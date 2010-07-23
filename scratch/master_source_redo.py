import numpy as np
import nipy.core.api as ni_api

from xipy.slicing import xipy_ras

from xipy.colors.rgba_blending import BlendedImages
from xipy.colors.rgba_blending import quick_convert_rgba_to_vtk

import xipy.colors.mayavi_tools as mt

from enthought.mayavi import mlab
from enthought.tvtk.api import tvtk

x, y, z = np.ogrid[-1:1:100j, -1:1:100j, -1:1:100j]
anat = x**2 + y**2 + z**2
activation = -np.sqrt(anat)
anat *= np.sin(4*x)*np.sin(4*y)*np.sin(4*z)
activation[activation < -.5] = -.6
activation -= activation.min()

main_img = ni_api.Image(
    anat,
    ni_api.AffineTransform.from_start_step('ijk', xipy_ras, [0,0,0], [1,1,1])
    )

over_img = ni_api.Image(
    activation,
    ni_api.AffineTransform.from_start_step('ijk', xipy_ras, [0,0,0], [1,1,1])
    )

bi = BlendedImages(vtk_order=True)
bi.main = main_img
bi.over = over_img

pd = tvtk.PointData()

class PDataDB(object):
    
    def __init__(self, pd):
        self.pd = pd
        self.db = dict()

    def add_to_pdata(self, array, name):
        if len(array.shape) > 3:
            vtk_order = quick_convert_rgba_to_vtk(array)
            flat_shape = (np.prod(vtk_order.shape[:3]), vtk_order.shape[3])
        else:
            vtk_order = array
            flat_shape = (np.prod(vtk_order.shape),)
        self.db[name] = vtk_order.shape
        arr = self.pd.get_array(name)
        if arr is None:
            n = self.pd.add_array(vtk_order.reshape(flat_shape))
            self.pd.get_array(n).name = name
        else:
            arr.from_array(vtk_order.reshape(flat_shape))

    def retrieve_vtk_volume(self, name):
        arr = self.pd.get_array(name)
        if arr is None:
            return None
        arr = arr.to_array()
        arr.shape = self.db[name]
        return arr

source = PDataDB(pd)

source.add_to_pdata(bi.main_rgba, 'main_colors')
source.add_to_pdata(bi.blended_rgba, 'blended_colors')
source.add_to_pdata(bi.over_rgba, 'over_colors')

# some random stuff
arr1 = np.random.randn(30,40,20)
source.add_to_pdata(arr1, 'random_normal')
arr2 = np.sin(2*np.pi*np.random.rand(30,40,20))
source.add_to_pdata(arr2, 'sine_random')

surf_source = mlab.pipeline.scalar_field(activation,
                                         transpose_input_array=False)
contour = mlab.pipeline.contour(surf_source)
contour_pd = contour.outputs[0]


src1 = mt.ArraySourceRGBA(transpose_input_array=False)
src1.scalar_data = source.retrieve_vtk_volume('main_colors')
src1.origin = surf_source.origin
src1.spacing = surf_source.spacing
n = src1.image_data.point_data.add_array(pd.get_array('blended_colors'))
src1.image_data.point_data.get_array(n).name = 'blended_colors'
n = src1.image_data.point_data.add_array(pd.get_array('over_colors'))
src1.image_data.point_data.get_array(n).name = 'over_colors'

## src2 = mt.ArraySourceRGBA(transpose_input_array=False)

sampler = tvtk.ProbeFilter()
## sampler.input = contour_pd

sampler.source = src1.image_data

surf_colors = mlab.pipeline.add_dataset(sampler.output)
surface = mlab.pipeline.surface(surf_colors)
