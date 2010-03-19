
class _OverlayRegistry(object):
    __reg_dict = dict()

    def add_overlay_plugin(self, overlay_name, window_class):
        self.__reg_dict[overlay_name] = window_class

    def names(self):
        return self.__reg_dict.keys()

    def items(self):
        return self.__reg_dict.items()

_registry = _OverlayRegistry()

def register_overlay_plugin(overlay_name, window_class):
    """Add an overlay class to the overlay plugin registry

    Parameters
    ----------
    overlay_name : str
        a name to identify the overlay class
    overlay_class : OverlayWindowInterface
        the window interface class
    """
    _registry.add_overlay_plugin(overlay_name, window_class)

def all_registered_plugins():
    """Returns all (name, class) pairs in the overlay plugin registry
    """
    return _registry.items()

# register XIPY plugin(s)
from xipy.overlay.image_overlay import ImageOverlayWindow
register_overlay_plugin('Image Overlay', ImageOverlayWindow)
