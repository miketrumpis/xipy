from xipy import TEMPLATE_MRI_PATH
from xipy.vis.three_dee_viewer import MayaviViewer
from xipy.overlay.image_overlay import ImageOverlayManager, ImageOverlayWindow
import xipy.io as xio

from PyQt4 import QtCore, QtGui
import sys, os
if QtGui.QApplication.startingUp():
    app = QtGui.QApplication(sys.argv)
else:
    app = QtGui.QApplication.instance() 

anat = xio.load_spatial_image(TEMPLATE_MRI_PATH)
func = xio.load_image(os.path.join(os.path.dirname(__file__),
                                  '../data/dtk_dti_out/dti_fa.nii'))
func_man = ImageOverlayManager(None, overlay=func)
#                              ^^^^ API will change soon 
        
win = MayaviViewer(image=anat)
win.make_tool_from_functional_manager(ImageOverlayWindow, func_man)

win.show()
app.exec_()
