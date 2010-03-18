from nipy.io.api import load_image
load_image('map_img.nii')
map = load_image('map_img.nii')
map.coordmap 
map.coordmap.affine
import nipy.core.api as ni_api
from nipy.core.reference.coordinate_map import reorder_output, compose
cmap_xyz = reorder_output(map.coordmap, 'xyz')
cmap_xyz.affine
map.coordmap.affine


from xipy.overlay import image_overlay
from xipy.slicing import load_resampled_slicer
from xipy import TEMPLATE_MRI_PATH
anat = load_resampled_slicer(TEMPLATE_MRI_PATH)
map_sl = load_resampled_slicer('map_img.nii')
oman = image_overlay.ImageOverlayManager(anat.bbox, overlay=map_sl)

oman.edit_traits()
