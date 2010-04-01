from PyQt4 import QtGui, QtCore

class TopLevelAuxiliaryWindow(QtGui.QWidget):
    """This is a specialized widget that can act like a top level window,
    but whose visibility can be toggled (much like a PyQt4 QDockWidget).
    The QAction object which controls the toggling is created by
    toggle_view_action(), just like a QDockWidget. These objects can be
    instantiated with a true parent widget (parent=mywidget), in which case
    it should not be top-level. Typically it would be instantiated with a
    reference to a widget (main_ref=mywidget) that should be considered
    a parent for whatever API access may be required.
    """
    
    notify_visible = QtCore.pyqtSignal(bool)
    notify_closed = QtCore.pyqtSignal()
    _activated = False
    def __init__(self, window_name='', parent=None, main_ref=None):
        if parent is not None or (main_ref is None and parent is None):
            QtGui.QWidget.__init__(self, parent)
            self.setParent(parent)
            self.padre = self.topLevelWidget()
        else:
            QtGui.QWidget.__init__(self)
            self.padre = main_ref
##             self.padre.destroyed.connect(self._destroy_aux)
##         app = QtCore.QCoreApplication.instance()
##         QtCore.QObject.connect(app,
##                                QtCore.SIGNAL('aboutToQuit'),
##                                self._destroy_aux)

        QtGui.QWidget.setSizePolicy(self,
                                    QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Expanding)
        self.updateGeometry()
        self.setObjectName(window_name)

    def toggle_view_action(self, visible=True):
        act = QtGui.QAction(self)
        act.setText(QtGui.QApplication.translate(
            self.padre.objectName(), self.objectName(),
            None,
            QtGui.QApplication.UnicodeUTF8
            ))
        QtCore.QObject.connect(
            act,
            QtCore.SIGNAL('toggled(bool)'),
            self._change_visibility
            )
        QtCore.QObject.connect(
            self,
            QtCore.SIGNAL('notify_visible(bool)'),
            act.setChecked
            )
        act.setCheckable(True)
        act.setChecked(visible)
        return act

    def activate(self):
        """ Prototype--this function is called when widget is set to visible
        """
        pass

    def deactivate(self):
        """ Prototype--this function is called when widget is set to invisible
        """
        pass
    
    def _change_visibility(self, b):
        self.setVisible(b)
        self.notify_visible.emit(b)
        if b:
            self.activate()
            self.raise_()
        else:
            self.deactivate()

    def closeEvent(self, ev):
        was_visible = self.isVisible()
        self._change_visibility(False)
        if was_visible:
            print 'ignoring event'
            ev.ignore()
            return

    def _destroy_aux(self):
        print 'sibling window closed'
        self.notify_closed.emit()
        #self.destroy(destroyWindow=True, destroySubWindows=True)
        self._change_visibility(False)
        self.close()
