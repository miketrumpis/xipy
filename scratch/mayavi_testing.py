from xipy.vis import mayavi_widgets
from xipy.vis import mayavi_tools
from xipy.overlay.image_overlay import ImageOverlayManager
from xipy.slicing import load_resampled_slicer
from xipy import TEMPLATE_MRI_PATH

from PyQt4 import QtCore, QtGui
import sys
if QtGui.QApplication.startingUp():
    app = QtGui.QApplication(sys.argv)
else:
    app = QtGui.QApplication.instance()

mainwin = QtGui.QMainWindow()
from nipy.io.api import load_image
anat = load_image(TEMPLATE_MRI_PATH)
func = load_image('map_img.nii')


win = mayavi_widgets.MayaviWidget(parent=mainwin)
win.mr_vis.blender.main = anat

## func_man = ImageOverlayManager(win.mr_vis.blender.bbox, overlay=func)

## win.add_toolbar(func_man)

my_track_file = '/Users/mike/workywork/dipy-vis/brain1/brain1_scan1_fiber_track_mni.trk'
from mini_track_control import mini_track_feature
mf = mini_track_feature(my_track_file, win.mr_vis)


## SOME QT4 SETUP
mainwin.setCentralWidget(win)
dock = QtGui.QDockWidget("Track Tool", mainwin)
dock.setWidget(mf.edit_traits(kind='subpanel', parent=dock).control)
mainwin.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
mainwin.show()
app.exec_()
## mf.edit_traits()
