from PyQt4 import QtGui, QtCore

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
