# PyQt4 library
from PyQt4 import QtGui, QtCore

# Enthought library
from enthought.traits.api import HasTraits

# XIPY imports
from xipy.vis.qt4_widgets.auxiliary_window import TopLevelAuxiliaryWindow

class VisualComponent(HasTraits):
    # NOTE! IT IS VERY IMPORTANT TO SET THIS UP AS A CONCRETE
    # INSTANCE BEFORE CREATING ANY SUBCLASSES
##     display = Instance('xipy.vis.mayavi_widgets.OrthoView3D')
    pass

class MayaviWidget(TopLevelAuxiliaryWindow):

    def __init__(self, parent=None, main_ref=None, functional_manager=None,
                 **traits):
        TopLevelAuxiliaryWindow.__init__(self,
                                         parent=parent,
                                         main_ref=main_ref)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        if functional_manager:
            traits['func_man'] = functional_manager
        from xipy.vis.mayavi_widgets.ortho_viewer_3d import OrthoViewer3D
        self.mr_vis = OrthoViewer3D(**traits)
        layout.addWidget(self.mr_vis.edit_traits(parent=self,
                                                 kind='subpanel').control)
        self.func_widget = None
        self.layout_box = layout
        if functional_manager is not None:
            self.add_toolbar(functional_manager)
        self.setObjectName('3D Plot')

    def add_toolbar(self, functional_manager):
        self.mr_vis.func_man = functional_manager
##         if self.func_widget is not None:
##             print 'removing old func widget'
##             self.layout_box.removeWidget(self.func_widget)
##             self.func_widget.close()
##         else:
##             print 'not removing old func widget'
##         self.func_widget = functional_manager.make_panel(parent=self)
##         self.layout_box.addWidget(self.func_widget)
##         self.update()
