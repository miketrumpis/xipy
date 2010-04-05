## THIS IS A HACK UP OF THE nipy.algorithms.interpolation MODULE, ALLOWING
## THE USER TO DISABLE MEMMAPPING, AND SET THE FILL VALUE, ETC
"""
Image interpolators using ndimage.
"""

__docformat__ = 'restructuredtext'

import os
import tempfile

import numpy as np

from scipy import ndimage


class ImageInterpolator(object):
    """
    Interpolate Image instance at arbitrary points in world space
    
    The resampling is done with scipy.ndimage.
    """

    def __init__(self, image, order=3, use_mmap=False):
        """
        Parameters
        ----------
        image : Image
           Image to be interpolated
        order : int
           order of spline interpolation as used in scipy.ndimage
        """
        self.image = image
        self.order = order
        self._datafile = None
        self._buildknots(use_mmap)

    def _buildknots(self, use_mmap):
        if self.order > 1:
##             data = ndimage.spline_filter(
##                 np.nan_to_num(np.asarray(self.image).astype('d')),
##                 self.order)
            in_data = np.asarray(self.image)
            data = ndimage.spline_filter(in_data,
                                         order=self.order,
                                         output=in_data.dtype)
        else:
##             data = np.nan_to_num(np.asarray(self.image).astype('d'))
            data = np.asarray(self.image)

        if use_mmap:
            if self._datafile is None:
                _, fname = tempfile.mkstemp()
                self._datafile = file(fname, mode='wb')
            else:
                self._datafile = file(self._datafile.name, 'wb')

            data.tofile(self._datafile)
            datashape = data.shape
            dtype = data.dtype
            del(data)
            self._datafile.close()
            self._datafile = file(self._datafile.name)
            self.data = np.memmap(self._datafile.name, dtype=dtype,
                                  mode='r+', shape=datashape)
        else:
            self.data = data

    def __del__(self):
        if self._datafile:
            self._datafile.close()
            try:
                os.remove(self._datafile.name)
            except:
                pass

    def evaluate(self, points, **interp_kws):
        """
        Parameters
        ----------
        points : ndarray, shape (nx x ny x nz x R)
            values in self.image.coordmap.output_coords

        Returns
        -------
        V: ndarray
           interpolator of self.image evaluated at points
        """
##         points = np.array(points, np.float64)
##         output_shape = points.shape[:-1]
##         points.shape = (np.product(output_shape), points.shape[-1])
        voxels = self.image.coordmap.inverse(points).T
        V = ndimage.map_coordinates(self.data,
                                    voxels,
                                    order=self.order,
                                    prefilter=False,
                                    output=self.data.dtype,
                                    **interp_kws)
        # ndimage.map_coordinates returns a flat array,
        # it needs to be reshaped to the original shape
##         V.shape = output_shape
        return V
