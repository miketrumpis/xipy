from PyQt4.QtGui import QIcon
from PyQt4.QtDesigner import QPyDesignerCustomWidgetPlugin
import os
from matplotlib import rcParams
from xipy.vis.qt4_widgets.ortho_slices import MplQT4OrthoSlicesWidget
rcParams['font.size'] = 9
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

class MplQT4OrthoSlicesPlugin(QPyDesignerCustomWidgetPlugin):
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
        return MplQT4OrthoSlicesWidget(parent=parent)

    def name(self):
        return "MplQT4OrthoSlicesWidget"

    def group(self):
        return "XIPY"

    def icon(self):
        image = os.path.join(rcParams['datapath'], 'images', 'matplotlib.png')
        return QIcon(image)

    def toolTip(self):
        return ""

    def whatsThis(self):
        return ""

    def isContainer(self):
        return False

    def domXml(self):
        return '<widget class="MplQT4OrthoSlicesWidget" name="mplfigwidget">\n' \
               '</widget>\n'

    def includeFile(self):
        return "xipy.vis.qt4_widgets.ortho_slices"


if __name__ == '__main__':
    import sys
    from PyQt4.QtGui import QApplication
    app = QApplication(sys.argv)
    widget = MplQT4OrthoSlicesWidget(parent=None)
    widget.show()
    sys.exit(app.exec_())

