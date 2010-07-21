import numpy as np
from PyQt4 import QtGui, QtCore

import matplotlib as mpl
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
## import matplotlib.cm as cm
import xipy.colors.color_mapping as cm

# XYZ: SUCKY CLASS NEEDS SERIOUS WORK

## class ColorbarPanel(QtGui.QWidget):
class ColorbarPanel(Canvas):
    """A small panel holding an MPL colorbar
    """
    def __init__(self, cmap=None, norm=None, parent=None, **kwargs):
##         QtGui.QWidget.__init__(self, parent)
##         self.setParent(parent)
        fig = Figure(figsize=kwargs.get('figsize', (2,1)),
                          dpi=kwargs.get('dpi', 100))
##         self.fig.canvas = Canvas(self.fig)
##         self.fig.canvas.setParent(self)
        self.ax = fig.add_axes([.1, .2, .8, .6])
        self.cb = self._create_colorbar(cmap, norm)
        self.t_pos = None
        Canvas.__init__(self, fig)
        Canvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding,
                             QtGui.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
        self.draw_idle()

    def _create_colorbar(self, cmap, norm):
        cmap = cmap or cm.gray
        norm = norm or mpl.colors.normalize(0,0)
        ticks = np.linspace(norm.vmin, norm.vmax, 7)
        self.ax.clear()
        cb = mpl.colorbar.ColorbarBase(self.ax, cmap=cmap, norm=norm,
                                       orientation='horizontal',
                                       ticks=ticks,
                                       format='%1.3f')
        for xt,_,_ in self.ax.xaxis.iter_ticks():
            xt.label.set_fontsize(9.0)
        return cb
        
    
    def _mark_threshold(self):
        if self.t_pos is None:
            return
        try:
            self.cb.lines.remove()
        except:
            pass
        self.t_pos = np.clip(self.t_pos, *self.cb.get_clim())
        self.cb.add_lines([self.t_pos], ['k'], [2.0])
        self.draw_idle()

    def __call__(self, x):
        rgb = self.cb.to_rgba(x)
        if len(rgb) < 2:
            rgb = rgb[0]
        rgb = rgb[:3]
        return mpl.colors.rgb2hex(rgb)
    
    def initialize(self):
        self.cb = self._create_colorbar(None, None)
        self.draw_idle()
    
    def add_cbar(self, cmap, norm, t_pos=None):
        self.cb = self._create_colorbar(cmap, norm)
        if t_pos is not None:
            self.t_pos = t_pos
            self._mark_threshold()
        else:
            self.draw_idle()

    def change_cmap(self, cmap):
        if str(cmap) in cm.cmap_d:
            cmap = cm.cmap_d[str(cmap)]
        norm = mpl.colors.normalize(*self.cb.get_clim())
        self.add_cbar(cmap, norm, t_pos=self.t_pos)

    def change_norm(self, norm):
        if type(norm) in (tuple, list):
            norm = mpl.colors.normalize(norm)
        cmap = self.cb.cmap
        self.add_cbar(cmap, norm, t_pos=self.t_pos)

    def change_threshold(self, t_pos):
        self.t_pos = t_pos
        self._mark_threshold()
        

#===============================================================================
#   Example
#===============================================================================
if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui, QtCore
    import numpy as np
    
    class ApplicationWindow(QtGui.QMainWindow):
        def __init__(self):
            QtGui.QMainWindow.__init__(self)
            mn = -10; mx = 10
            w = QtGui.QWidget(parent=self)
            self.setCentralWidget(w)
            self.cmap = ColorbarPanel(parent=w)
            self.cmap.add_cbar(cm.hot, mpl.colors.Normalize(mn,mx))
            
            self.slider = QtGui.QSlider(parent=w)
            self.slider.setRange(mn, mx)
            self.slider.valueChanged.connect(self._update_cmap)
            self.slider.setOrientation(0x1)

            vbox = QtGui.QVBoxLayout(w)
            vbox.addWidget(self.cmap)
            vbox.addWidget(self.slider)
            
            
        def _update_cmap(self):
            self.cmap.change_threshold(self.slider.value())

    if QtGui.QApplication.startingUp():
        app = QtGui.QApplication(sys.argv)
    else:
        app = QtGui.QApplication.instance()
    win = ApplicationWindow()
    win.show()
    sys.exit(app.exec_())
