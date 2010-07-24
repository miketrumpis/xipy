from PyQt4 import QtGui
import numpy as np

from xipy.overlay.interface import OverlayWindowInterface, OverlayInterface
from xipy.slicing import xipy_ras

import nipy.core.api as ni_api

# important to import XIPY stuff before traits UI stuff
import enthought.traits.api as t
import enthought.traits.ui.api as tui

class SimpleWindow(OverlayWindowInterface):
    tool_name = 'Random Plotter'

    # little bit of boiler plate setup
    def __init__(self, *args, **kwargs):
        OverlayWindowInterface.__init__(self, *args, **kwargs)
        if self.func_man is None:
            self.func_man = SimpleManager(
                image_signal=self.image_changed,
                loc_signal=self.loc_changed,
                props_signal=self.image_props_changed)
        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.func_man.make_panel(parent=self))
        QtGui.QWidget.setSizePolicy(self,
                                    QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Expanding)
        self.updateGeometry()

class SimpleManager(OverlayInterface):

    new_plot = t.Button('Push new data')

    def _new_plot_fired(self):
        self._push_random_data()

    def _push_random_data(self):
        new_data = np.random.randn(30,30,30)
        affine = np.diag([2,2,2,1])
        affine[:3,-1] = -30, -30, -30
        coordmap = ni_api.AffineTransform.from_params(
            'ijk', xipy_ras, affine
            )
        image = ni_api.Image(new_data, coordmap)
        self.notify_image_change(image)

    def notify_image_change(self, image):
        self.overlay = image
        self.overlay_updated = True
        if self.image_signal:
            self.image_signal.emit(self)
        
    view = tui.View(
        tui.HGroup(
            tui.Item('new_plot', show_label=False),
            tui.Include('image_props_group')
            ),
        resizable=True
        )


def main():
    app = QtGui.QApplication(sys.argv)

    # register the nutmeg plugin before launching the ortho viewer
    from xipy.overlay.plugins import register_overlay_plugin, \
         all_registered_plugins
    register_overlay_plugin(SimpleWindow)
    from xipy.vis.three_dee_viewer import MayaviViewer
    from xipy import TEMPLATE_MRI_PATH
    win = MayaviViewer(image=TEMPLATE_MRI_PATH)
    win.show()
    return win, app
    

if __name__=='__main__':
    import sys
    win, app = main()
    sys.exit(app.exec_())
    
