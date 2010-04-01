"""This module contains some modifications of Matplotlib colormapping code
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cbook as cbook
from matplotlib._cm import datad
import numpy as np
import numpy.ma as ma

parts = np.__version__.split('.')
NP_MAJOR, NP_MINOR = map(int, parts[:2])
# true if clip supports the out kwarg
NP_CLIP_OUT = NP_MAJOR>=1 and NP_MINOR>=2

class MixedAlphaColormap(LinearSegmentedColormap):

    def __call__(self, X, alpha=1.0, bytes=False):
        """
        *X* is either a scalar or an array (of any dimension).
        If scalar, a tuple of rgba values is returned, otherwise
        an array with the new shape = oldshape+(4,). If the X-values
        are integers, then they are used as indices into the array.
        If they are floating point, then they must be in the
        interval (0.0, 1.0).
        Alpha must be a scalar.
        If bytes is False, the rgba values will be floats on a
        0-1 scale; if True, they will be uint8, 0-255.
        """

        if not self._isinit: self._init()
        if cbook.iterable(alpha):
            if len(alpha) != self.N:
                raise ValueError('Provided alpha LUT is not the right length')
            alpha = np.clip(alpha, 0, 1)
            # repeat the last alpha value for i_under, i_over
            alpha = np.r_[alpha, alpha[-1], alpha[-1]]
        else:
            alpha = min(alpha, 1.0) # alpha must be between 0 and 1
            alpha = max(alpha, 0.0)

        self._lut[:-1,-1] = alpha  # Don't assign global alpha to i_bad;
                                   # it would defeat the purpose of the
                                   # default behavior, which is to not
                                   # show anything where data are missing.
        mask_bad = None
        if not cbook.iterable(X):
            vtype = 'scalar'
            xa = np.array([X])
        else:
            vtype = 'array'
            # force a copy here -- the ma.array and filled functions
            # do force a cop of the data by default - JDH
            xma = ma.array(X, copy=True)
            xa = xma.filled(0)
            mask_bad = ma.getmask(xma)
        if xa.dtype.char in np.typecodes['Float']:
            np.putmask(xa, xa==1.0, 0.9999999) #Treat 1.0 as slightly less than 1.
            # The following clip is fast, and prevents possible
            # conversion of large positive values to negative integers.

            if NP_CLIP_OUT:
                np.clip(xa * self.N, -1, self.N, out=xa)
            else:
                xa = np.clip(xa * self.N, -1, self.N)
            xa = xa.astype(int)
        # Set the over-range indices before the under-range;
        # otherwise the under-range values get converted to over-range.
        np.putmask(xa, xa>self.N-1, self._i_over)
        np.putmask(xa, xa<0, self._i_under)
        if mask_bad is not None and mask_bad.shape == xa.shape:
            np.putmask(xa, mask_bad, self._i_bad)
        if bytes:
            lut = (self._lut * 255).astype(np.uint8)
        else:
            lut = self._lut
        rgba = np.empty(shape=xa.shape+(4,), dtype=lut.dtype)
        lut.take(xa, axis=0, mode='clip', out=rgba)
                    #  twice as fast as lut[xa];
                    #  using the clip or wrap mode and providing an
                    #  output array speeds it up a little more.
        if vtype == 'scalar':
            rgba = tuple(rgba[0,:])
        return rgba


cmap_d = dict()

# reverse all the colormaps.
# reversed colormaps have '_r' appended to the name.

def _reverser(f):
    def freversed(x):
        return f(1-x)
    return freversed

def revcmap(data):
    data_r = {}
    for key, val in data.iteritems():
        if callable(val):
            valnew = _reverser(val)
                # This doesn't work: lambda x: val(1-x)
                # The same "val" (the first one) is used
                # each time, so the colors are identical
                # and the result is shades of gray.
        else:
            valnew = [(1.0 - a, b, c) for a, b, c in reversed(val)]
        data_r[key] = valnew
    return data_r

LUTSIZE = mpl.rcParams['image.lut']

_cmapnames = datad.keys()  # need this list because datad is changed in loop

for cmapname in _cmapnames:
    cmapname_r = cmapname+'_r'
    cmapspec = datad[cmapname]
    if 'red' in cmapspec:
        datad[cmapname_r] = revcmap(cmapspec)
        cmap_d[cmapname] = MixedAlphaColormap(
                                cmapname, cmapspec, LUTSIZE)
        cmap_d[cmapname_r] = MixedAlphaColormap(
                                cmapname_r, datad[cmapname_r], LUTSIZE)
    else:
        revspec = list(reversed(cmapspec))
        if len(revspec[0]) == 2:    # e.g., (1, (1.0, 0.0, 1.0))
            revspec = [(1.0 - a, b) for a, b in revspec]
        datad[cmapname_r] = revspec

        cmap_d[cmapname] = MixedAlphaColormap.from_list(
                                cmapname, cmapspec, LUTSIZE)
        cmap_d[cmapname_r] = MixedAlphaColormap.from_list(
                                cmapname_r, revspec, LUTSIZE)

locals().update(cmap_d)
