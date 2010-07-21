# PyQt4 library
from PyQt4 import QtGui, QtCore

# NumPy
import numpy as np

# Enthought library
import enthought.traits.api as t

# XIPY imports
from xipy.vis.qt4_widgets.auxiliary_window import TopLevelAuxiliaryWindow

class VisualComponent(t.HasTraits):
    name = t.String
    display = t.Instance(
        'xipy.vis.mayavi_widgets.ortho_viewer_3d.OrthoViewer3D'
        )

class MayaviWidget(TopLevelAuxiliaryWindow):

    def __init__(self, parent=None, main_ref=None,
                 functional_manager=None, manage_overlay=True,
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
        self.mr_vis = OrthoViewer3D(manage_overlay=manage_overlay, **traits)
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


from xipy.colors.mayavi_tools import ArraySourceRGBA
from xipy.colors.rgba_blending import BlendedImages, quick_convert_rgba_to_vtk
class MasterSource(ArraySourceRGBA):
    """
    This class monitors a BlendedImages object and sets up image
    channels for the main image, over image, and blended image.

    It may have additional channels
    """

    # XXX: This source should have a signal saying when a channel may
    # disappear!!! EG, if blender.over gets set to None, and some module
    # is visualizing "over_colors", then there will be a big fat crash!!
    
    blender = t.Instance(BlendedImages)
    over_channel = 'over_colors'
    main_channel = 'main_colors'
    blended_channel = 'blended_colors'

    rgba_channels = t.Property
    all_channels = t.Property

    transpose_input_array = False

    @t.on_trait_change('transpose_input_array')
    def _ignore_transpose(self):
        self.trait_setq(transpose_input_array=False)

    @t.on_trait_change('blender')
    def _check_vtk_order(self):
        if not self.blender.vtk_order:
            raise ValueError('BlendedImages instance must be in VTK order')

    def _get_all_channels(self):
        pdata = self.image_data.point_data
        names = [pdata.get_array_name(n)
                 for n in xrange(pdata.number_of_arrays)]
        return names

    def _get_rgba_channels(self):
        primary_channels = (self.over_channel,
                            self.main_channel,
                            self.blended_channel)
        names = self.all_channels
        return [n for n in names if n in primary_channels]

    ## The convention will be to have main_rgba be the primary array
    ## in scalar_data. If main_rgba isn't present, then set it to
    ## over_rgba (if present)

    def _set_primary_scalars(self, arr, name):
        if self.scalar_data is not None \
               and self.scalar_data.size != arr.size:
            self.flush_arrays()
        rgba = quick_convert_rgba_to_vtk(arr)
        self.scalar_data = rgba
        self.scalar_name = name
        self.origin = self.blender.img_origin
        self.spacing = self.blender.img_spacing
        self.update()

    @t.on_trait_change('blender.main_rgba')
    def _set_main_array(self):
        # Set main_rgba into scalar_data.

        # changes
        # 1) from size 1 to size 1
        # 2) from size 1 to size 2
        # 3) from size 1 to 0 (with over_rgba)
        # 4) from size 1 to 0 (without over_rgba)
        # 5) from 0 to size 1


        print 'main rgba update', self.blender.main_rgba.size, self.blender.over_rgba.size
        # cases 1, 2, 5
        if self.blender.main_rgba.size:
            self._set_primary_scalars(self.blender.main_rgba, self.main_channel)

        # cases 3, 4 will be triggered if and when over_rgba changes

    @t.on_trait_change('blender.over_rgba')
    def _set_over_array(self):
        # Set over_rgba (and possibly blended_rgba) into appropriate arrays

        # cases
        # 1) main_rgba.size > 0
        # 2) main_rgba.size == 0
        print 'over rgba update', self.blender.over_rgba.size, self.blender.main_rgba.size
        if not self.blender.main_rgba.size and self.blender.over_rgba.size:
            self._set_primary_scalars(
                self.blender.over_rgba, self.over_channel
                )
            return
        if not self.blender.over_rgba.size:
            self.flush_arrays(names=[self.over_channel, self.blended_channel])
            return
        elif self.blender.over_rgba.size != self.blender.main_rgba.size:
            # this should always be prevented in the BlendedImages class
            raise RuntimeError('Color channel sizes do not match')

        # otherwise, append/set a new array with over_channel tag
        self.set_new_array(
            self.blender.over_rgba, self.over_channel, update=False
            )        
        # this obviously also changes the blended array
        self.set_new_array(
            self.blender.blended_rgba, self.blended_channel, update=True
            )

    def flush_arrays(self, names=[], update=True):
        pdata = self.image_data.point_data
        if not names:
            names = [pdata.get_array_name(n)
                     for n in xrange(pdata.number_of_arrays)]
        for n in names:
            pdata.remove_array(n)
        if update:
            self.image_data.update()
            self.image_data.update_traits()
            self.data_changed = True
            self.update()

    def set_new_array(self, arr, name, update=True):
        """

        Parameters
        ----------
        arr : 4-component ndarray with dtype = uint8
          The `arr` parameter must be shaped (npts x 4), or shaped
          (nz, ny, nx, 4) in C-order (like BlendedImage RGBA arrays when
          vtk_order is True)

        name : str
          name of the array
        
        """
        pdata = self.image_data.point_data
        if len(arr.shape) > 2:
            if len(arr.shape) > 3:
                flat_arr = arr.reshape(np.prod(arr.shape[:3]), 4)
            else:
                flat_arr = arr.ravel()
        else:
            flat_arr = arr
        chan = pdata.get_array(name)
        if chan:
            chan.from_array(flat_arr)
        else:
            n = pdata.add_array(flat_arr)
            pdata.get_array(n).name = name
        if update:
            self.image_data.update()
            # this one definitely needed
            self.pipeline_changed = True
            self.pipeline_changed = True

    
        
        
        
        
