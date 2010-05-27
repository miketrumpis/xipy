from dipy.io import trackvis as tv
from dipy.core import track_performance as tp
from dipy.core import track_metrics as tm
import nipy.core.api as ni_api
import numpy as np
import os, sys

#example_brain = '/Users/mike/workywork/dipy-vis/brain1/brain1_scan1_fiber_track_mni.trk'

def load_reduce_translate(fname, reduction=1):
    """ Loads a trackvis file using DIPY io, only keeping a ratio of
    1/reduction tracks. Performs trajectory approximation to reduce the
    track lengths.

    Parameter
    ---------
    fname : str
        The Trackvis file
    reduction : int, optional
        The reduction factor (keep only 1 line per set of reduction lines)

    Returns
    -------
    tracks : list
        A list of tracks
    """

    lines, hdr = tv.read(fname)
    print 'loaded,',
    sys.stdout.flush()

    #ras = tv.get_affine(hdr)
    ras = tv.aff_from_hdr(hdr)
    if not ras[:3,-1].any():
        # dot(ras[:3,:3],md_vox) + t = (0,0,0) --> t = -dot(ras[:3,:3],md_vox)
        md_vox = hdr['dim']/2
        t = -np.dot(ras[:3,:3], md_vox)
        ras[:3,-1] = t

    tracks = [l[0] for l in lines[::reduction]]
    del lines
    ras_cmap = ni_api.AffineTransform.from_params('ijk', 'xyz', ras)
    flat_tracks, breaks = flatten_tracks(tracks)
    del tracks
    flat_tracks_xyz = ras_cmap(flat_tracks)
    tracks = recombine_flattened_tracks(flat_tracks_xyz, breaks)
    
    print 'translated,',
    sys.stdout.flush()
    
    tracks_reduced = [tp.approximate_ei_trajectory(line, alpha=np.pi/4)
                      for line in tracks ]

    print 'reduced,'
    sys.stdout.flush()
    return tracks_reduced

def flatten_tracks(tracks):
    breaks = np.array([ len(line) for line in tracks ])
    ncoords = tracks[0].shape[-1]
    flat_pts = np.empty((breaks.sum(), ncoords), tracks[0].dtype)
    pc = 0
    for n in xrange(len(tracks)):
        flat_pts[pc:pc+breaks[n]] = tracks[n]
        pc += breaks[n]

    return flat_pts, breaks

def recombine_flattened_tracks(flat_tracks, breaks):
    tracks = []
    pc = 0
    for line_len in breaks:
        tracks.append( flat_tracks[pc:pc+line_len] )
        pc += line_len
    return tracks

def simple_directional_colors(tracks):
    """ Create a color for each track that indicates the mean orientation
    of the track. Do this by finding the mean orientation of all tracks.
    Convert these orientation coordinates to RGB coordinates by taking
    the absolute value of all coordinates, and normalizing relative to the
    maximal coordinates among all track orientations.

    Parameters
    ---------
    tracks : a list of track lines

    Returns
    -------
    rgb_colors : ndarray
        a (len(tracks) x 3) array of RGB color values in [0,1]
    """
    mn_orient = [tm.mean_orientation(line) for line in tracks]
    rgb_vecs = np.abs(mn_orient)
    rgb_vecs /= rgb_vecs.max(axis=0)
    return rgb_vecs
    
    

