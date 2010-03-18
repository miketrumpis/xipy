from nutmeg.vis import *
from nutmeg.vis import single_slice_plot as ssp
from nutmeg.vis.volume_utils import limits_to_extents
from nutmeg.vis.tfbeam_manager import SignalBeamManager
from nutmeg.vis.overlays import overlay_panel_factory

import os
import numpy as np
from PyQt4 import QtGui
from PyQt4 import QtCore
import matplotlib as mpl
mpl.use('QT4Agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
from matplotlib.backend_bases import LocationEvent
from matplotlib.figure import Figure
import matplotlib.cm as cm

BLITTING=True

def browse_files(parent, dialog='Select File', wildcard=''):
    fname = QtGui.QFileDialog.getOpenFileName(parent,
                                              dialog,
                                              QtCore.QDir.currentPath(),
                                              wildcard
                                              )
    return str(fname)

def browse_multiple_files(parent, dialog='Select File(s)', wildcard=''):
    fnames = QtGui.QFileDialog.getOpenFileNames(parent,
                                                dialog,
                                                QtCore.QDir.currentPath(),
                                                wildcard
                                                )
    return [str(fn) for fn in fnames]

class FakeMPLEvent(object):
    def __init__(self, c, a, x, y):
        self.canvas = c
        self.inaxes = a
        self.xdata = x
        self.ydata = y



#===============================================================================
#   Example
#===============================================================================
if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui, QtCore
    #from numpy import linspace
    import numpy as np
    
    class ApplicationWindow(QtGui.QMainWindow):
        def __init__(self):
            QtGui.QMainWindow.__init__(self)
##             centralwidget = QtGui.QWidget(self)
##             self.setCentralWidget(centralwidget)
##             vlayout = QtGui.QVBoxLayout(centralwidget)
            self.mplwidget = MplQT4OrthoSlicesWidget(parent=self)
            self.setCentralWidget(self.mplwidget)
##             vlayout.addWidget(self.mplwidget)
            dock = QtGui.QDockWidget('tf_window', self)
            dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.BottomDockWidgetArea)
            self.tfwidget = MplQT4TimeFreqWindow((), (), (), parent=self)
            dock.setWidget(self.tfwidget)
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
            #self.tfwidget.show()
            #self.tfwidget.activateWindow()
            #self.tfwidget.raise_()
            #vlayout.addWidget(self.tfwidget)
            self.plot()
            
        def plot(self):
            data = [np.random.randn(10,10) for x in [0,1,2]]
            self.mplwidget.initialize_plots(data,
                                            (0,0,0),
                                            [(-50,50)]*3,
                                            interpolation='nearest')
            self.mplwidget.draw()
                                       
            #self.mplwidget.fig.xlim = (-50,50)
            #self.mplwidget.fig.ylim = (-25,50)
##             self.mplwidget.fig.set_limits((-50,50,-25,50))


    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance()
    win = ApplicationWindow()
    win.show()
    sys.exit(app.exec_())
        
