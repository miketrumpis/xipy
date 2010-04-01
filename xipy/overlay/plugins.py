
class _OverlayRegistry(object):
    __reg_dict = dict()

    def add_overlay_plugin(self, overlay_name, window_class):
        self.__reg_dict[overlay_name] = window_class

    def names(self):
        return self.__reg_dict.keys()

    def items(self):
        return self.__reg_dict.items()

_registry = _OverlayRegistry()

def register_overlay_plugin(window_class, plugin_name=None):
    """Add an overlay class to the overlay plugin registry

    Parameters
    ----------
    overlay_class : OverlayWindowInterface
        the window interface class
    overlay_name : str, optional
        a name to identify the overlay class (by default, the tool_name
        from the window class is used)
    """
    used_name = plugin_name if plugin_name else window_class.tool_name
    _registry.add_overlay_plugin(used_name, window_class)

def all_registered_plugins():
    """Returns all (name, class) pairs in the overlay plugin registry
    """
    return _registry.items()

# register XIPY plugin(s)
from xipy.overlay.image_overlay import ImageOverlayWindow
register_overlay_plugin(ImageOverlayWindow)
