from PyQt4.QtGui import QIcon
from PyQt4.QtDesigner import QPyDesignerCustomWidgetPlugin
import os
from xipy.vis.mayavi_widgets import MayaviWidget

class MayaviWidgetPlugin(QPyDesignerCustomWidgetPlugin):
    def __init__(self, parent=None):
        QPyDesignerCustomWidgetPlugin.__init__(self)

        self._initialized = False

    def initialize(self, formEditor):
        if self._initialized:
            return

        self._initialized = True

    def isInitialized(self):
        return self._initialized

    def createWidget(self, parent):
        return MayaviWidget(parent=parent)

    def name(self):
        return "MayaviWidget"

    def group(self):
        return "XIPY"

##     def icon(self):
##         image = os.path.join(rcParams['datapath'], 'images', 'matplotlib.png')
##         return QIcon(image)

    def toolTip(self):
        return ""

    def whatsThis(self):
        return ""

    def isContainer(self):
        return False

    def domXml(self):
        return '<widget class="MayaviWidget" name="mayavi_widget">\n' \
               '</widget>\n'

    def includeFile(self):
        return "xipy.vis.mayavi_widgets"

if __name__=='__main__':
    from PyQt4 import QtCore, QtGui
    import sys
    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance() 

    win = MayaviWidget(parent=None)
    
    win.show()
    app.exec_()
