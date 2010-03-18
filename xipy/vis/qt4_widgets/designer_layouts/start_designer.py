#!/usr/bin/env python
import os

cwd = os.path.dirname(__file__)
plugin_path = os.path.join(cwd, 'plugin/python')

try:
    os.environ['PYQTDESIGNERPATH'] += os.path.pathsep+plugin_path
except KeyError:
    os.environ['PYQTDESIGNERPATH'] = os.path.pathsep+plugin_path
from PyQt4 import QtCore
des = QtCore.QProcess()
dbin = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.BinariesPath)
dbin += '/Designer.app/Contents/MacOS/Designer'
des.start(dbin)
des.waitForFinished(-1)


