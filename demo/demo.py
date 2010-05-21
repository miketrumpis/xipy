import numpy as np

from xipy.slicing import load_resampled_slicer
from xipy import TEMPLATE_MRI_PATH
from xipy.vis.three_dee_viewer import MayaviViewer
from xipy.overlay.image_overlay import ImageOverlayManager, ImageOverlayWindow
import xipy.volume_utils as vu

from PyQt4 import QtCore, QtGui
import sys, os
if QtGui.QApplication.startingUp():
    app = QtGui.QApplication(sys.argv)
else:
    app = QtGui.QApplication.instance() 

anat = load_resampled_slicer(TEMPLATE_MRI_PATH)
from nipy.io.api import load_image
import nipy.core.api as ni_api
img = load_image(os.path.join(os.path.dirname(__file__),
                              '../data/dtk_dti_out/dti_fa.nii'))
func = load_resampled_slicer(img)

func_man = ImageOverlayManager(anat.bbox, overlay=func)
        
win = MayaviViewer(image=anat)
oman = ImageOverlayManager(vu.world_limits(anat.raw_image),
                           overlay=func)
win.make_tool_from_functional_manager(ImageOverlayWindow, oman)

win.show()
app.exec_()
