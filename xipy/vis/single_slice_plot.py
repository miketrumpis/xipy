import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
import numpy as np

import xipy.vis.color_mapping as cm

now = True

# I am setting this globally?? why is there no easier way!
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

def try_or_pass(default=None):
    def dec(func):
        def to_run(*args, **kwargs):
            try:
                a = func(*args, **kwargs)
                if a is not None:
                    return a
            except:
                return default
        return to_run
    return dec

class SliceFigure(object):
    """ This is a "has-a" class that holds and manages a
    matplotlib.figure.Figure object with one Axes. The figure will be drawn
    in some graphical backend that is determined a priori (such that
    fig.canvas is already a set attribute).
    """
    img_num = 0
    def __init__(self, fig, limits, blit=True, px=0, py=0):
        self._blit = blit
        self.fig = fig
        self.canvas = fig.canvas
        self.bkgrnd = None
        self._noblit_list = []
        self._slice_images = []
        self.px, self.py = px, py
        if not fig.axes:
            self.ax = self.fig.add_subplot(111, aspect='equal',
                                           adjustable='box')
        else:
            self.ax = self.fig.axes[-1]
        self.set_limits(limits)
        self._init_crosshairs(px,py)
        self._saved_size = self.ax.bbox.size
    
    # axes properties:
    # xlim, ylim
    @try_or_pass()
    def _get_xlim(self):
        return self.ax.get_xlim()
    @try_or_pass()
    def _set_xlim(self, xlim):
        ax = self.ax
        ax.set_xlim(xlim)
        if ax.artists:
            row_line = ax.artists[0]
            row_line.set_data(self._crosshairs_data(self.px, self.py)[0])
        self.draw(save=True)
    @try_or_pass()
    def _get_ylim(self):
        return self.ax.get_ylim()
    @try_or_pass()
    def _set_ylim(self, ylim):
        ax = self.ax
        ax.set_ylim(ylim)
        if ax.artists:
            col_line = ax.artists[1]
            col_line.set_data(self._crosshairs_data(self.px, self.py)[1])
        self.draw(save=True)

    xlim = property(_get_xlim, _set_xlim)
    ylim = property(_get_ylim, _set_ylim)

    def _init_crosshairs(self, px, py):
        self.px, self.py = px,py
        row_data, col_data = self._crosshairs_data(px, py)
        row_line = Line2D(row_data[0], row_data[1],
                          color="r", linewidth=0.75, alpha=.5)
        col_line = Line2D(col_data[0], col_data[1],
                          color="r", linewidth=0.75, alpha=.5)
        self.crosshairs = (row_line, col_line)
        ax = self.ax
        ax.add_artist(row_line)
        ax.add_artist(col_line)
        self._noblit_list.append(row_line)
        self._noblit_list.append(col_line)

    def _crosshairs_data(self, px, py):
        ylim = self.ylim
        xlim = self.xlim
        data_wd, data_ht = (xlim[1]-xlim[0], ylim[1]-ylim[0])
        # make cross hairs the same length, based on the minimum extent
        xhair_len = min(data_wd, data_ht)/4.0
        row_data = ((px-xhair_len, px+xhair_len), (py, py))
        col_data = ((px, px), (py-xhair_len, py+xhair_len))
        return row_data, col_data

    def _draw_crosshairs(self):
        if hasattr(self, 'crosshairs') and self._blit:
            if self.bkgrnd is not None:
                self.canvas.restore_region(self.bkgrnd)
            self.ax.draw_artist(self.crosshairs[0])
            self.ax.draw_artist(self.crosshairs[1])
            self.canvas.blit(self.ax.bbox)
        else:
            self.draw(when=now)
            
    def set_limits(self, lims):
        self.xlim = lims[:2]
        self.ylim = lims[2:]

##     def set_image_extent(self, extent):
##         for im in self._slice_images:
##             im._extent = extent
##         self.set_limits(extent[:2], extent[2:])
    
    def move_crosshairs(self, px, py):
        # if event happens outside of axes, px and/or py may be None
        if px is not None: self.px = px
        if py is not None: self.py = py
        row_data, col_data = self._crosshairs_data(self.px, self.py)
        row_line, col_line = self.crosshairs
        row_line.set_data(*row_data)
        col_line.set_data(*col_data)
        self._draw_crosshairs()

    def toggle_crosshairs_visible(self, mode=True):
        for line in self.crosshairs:
            line.set_visible(mode)
        self.draw(when=now)
    
    def spawn_image(self, sl_data, loc=None, **img_kws):
        ax = self.ax
        ax.hold(True)
        if 'origin' not in img_kws:
            img_kws['origin'] = 'lower'
        if 'extent' not in img_kws:
            img_kws['extent'] = self.xlim + self.ylim
        if 'cmap' not in img_kws:
            # important to make SURE this is a home-grown colormap!
            img_kws['cmap'] = cm.jet
        img = AxesImage(ax, **img_kws)
        ax.images.append(img)
        s_img = SliceImage(self, img, sl_data)
        self._slice_images.append(s_img)
        if loc:
            self.move_crosshairs(*loc)
        return s_img

    def pop_image(self, s_img):
        try:
            idx = self._slice_images.index(s_img)
            self._slice_images.pop(idx)
            self.ax.images.pop(idx)
        except:
            pass

    def set_data(self, slice_list):
        # be lax if slice_list comes in as a non-nested list of arrays
        if len(self._slice_images)==1 and type(slice_list) not in (list,tuple):
            slice_list = [slice_list]
        for img, data in zip(self._slice_images, slice_list):
            img.set_data(data)

    def get_imageobj(self, num=-1):
        if num < 0:
            num = self._img_num
        images = self.ax.images
        return images[num] if len(images) > num else None

    def _savebbox(self):
        if not self._blit:
            return
        state = [artist.get_visible() for artist in self._noblit_list]
        if any(state):
            # should act NOW, but skip this section of the callback
            for s, artist in zip(state, self._noblit_list):
                if s:
                    artist.set_visible(False)
            self.draw(when=now)

        self.bkgrnd = self.canvas.copy_from_bbox(self.ax.bbox)
        if any(state):
            for s, artist in zip(state, self._noblit_list):
                if s:
                    artist.set_visible(True)

    def draw(self, when=not now, save=False):
        # if the axes have been resized, then force a save of the image
        if not (self.ax.bbox.size == self._saved_size).all():
            print 'forcing a save'
            self._saved_size = self.ax.bbox.size
            save = True
        if save:
            self._savebbox()
        if when:
            # do blocking draw
            self.canvas.draw()
        else:
            self.canvas.draw_idle()

## # XYZ: SHOULD ACTUALLY MAKE THIS AN AXESIMAGE SUBCLASS.. CONSIDER
## # OVER-RIDING to_rgba()

## class SliceImage(AxesImage):
##     # this is an optional lut for the alpha channel
##     alpha_lut = None

class SliceImage(object):
    """ This is a fancy container for a MPL AxesImage object, which can
    change up colormaps, interpolation modes, extents, and color scaling.
    """

    def __init__(self, fig, img, data):
        self.fig = fig # parent SliceFigure
        self.img = img # AxesImage
        self.data = data # ndarray
        if data.dtype.char != 'B' and len(data.shape) != 2:
            raise ValueError("data needs to have exactly two dimensions")
        self.img.set_data(data)
        self.data = data
        self.fig.draw(save=True, when=now)

    ############### GETTERS AND SETTERS FOR PROPERTIES #########################
    # image properties (can be set by user):
    # extent, cmap, norm, interpolation, alpha
    @try_or_pass()
    def _set_extent(self, extent):
        self.img.set_extent(extent)
        self.fig.draw(save=True)
    @try_or_pass()
    def _get_extent(self):
        return self.img.get_extent()
    @try_or_pass()
    def _set_cmap(self, cmap):
        self.img.set_cmap(cmap)
        self.fig.draw(save=True)
    @try_or_pass()
    def _get_cmap(self):
        return self.img.get_cmap()        
    @try_or_pass()
    def _set_interp(self, interp):
        self.img.set_interpolation(interp)
        self.fig.draw(save=True)
    @try_or_pass()
    def _get_interp(self):
        return self.img._interpolation
    @try_or_pass()
    def _set_norm(self, norm):
        self.img.set_norm(norm)
        self.fig.draw(save=True)
    @try_or_pass()
    def _get_norm(self):
        return self.img.norm
    @try_or_pass()
    def _set_alpha(self, alpha):
        self.img.set_data(self.data)
        self.img.set_alpha(alpha)
        self.fig.draw(save=True)
    @try_or_pass(default=1)
    def _get_alpha(self):
        return self.img.get_alpha()
    cmap = property(_get_cmap, _set_cmap)
    interp = property(_get_interp, _set_interp)
    norm = property(_get_norm, _set_norm)
    alpha = property(_get_alpha, _set_alpha)
    extent = property(_get_extent, _set_extent)    
        
    def set_properties(self, **props):
        if 'extent' in props:
            self.img.set_extent(props['extent'])
        if 'cmap' in props:
            self.img.set_cmap(props['cmap'])
        if 'interpolation' in props:
            self.img.set_interpolation(props['interpolation'])
        if 'norm' in props:
            self.img.set_norm(props['norm'])
        if 'alpha' in props:
            self.img.set_alpha(props['alpha'])
        self.fig.draw(save=True)
    
    def set_data(self, data):
        self.data = data
        self.img.set_data(data)
        self.fig.draw(save=True)
##         self.fig.draw(save=True, when=now)


    
if __name__=='__main__':
    import sys
    import matplotlib as mpl
    try:
        mpl.use(sys.argv[1])
    except:
        mpl.use('Qt4Agg')
    import matplotlib.figure
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    import matplotlib.pyplot as pp    
##     fig = mpl.figure.Figure()
##     fig.canvas = Canvas(fig)
    fig = pp.figure()
    sfig = SliceFigure(fig, [-50,50,-50,50], blit=True)
    img = sfig.spawn_image(np.random.randn(100,100))

    # connect a crosshair moving event
    def _move_xhair(ev):
        if not ev.inaxes:
            return
        sfig.move_crosshairs(ev.xdata, ev.ydata)

    # if clicking outside the axes, then run through a series of 20 images
    def _animate_images(ev):
        if ev.inaxes:
            return
        for x in xrange(20):
            img.set_data(np.random.randn(100,100))

    fig.canvas.mpl_connect('motion_notify_event', _move_xhair)
##     fig.canvas.mpl_connect('button_press_event', _animate_images)

    pp.show()
##     import cProfile, pstats
##     cProfile.runctx('pp.show()', globals(), locals(), 'sfigure.prof')
##     s = pstats.Stats('sfigure.prof')
##     s.strip_dirs().sort_stats('cumulative').print_stats()
    
