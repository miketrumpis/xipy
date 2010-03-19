from scratch import get_tracks
import numpy as np
from nose.tools import assert_true, assert_equal, assert_false

def test_flatten():
    ntracks = 100
    tracks = []
    for n in xrange(ntracks):
        tlen = np.random.randint(0,high=15)
        tracks.append( np.random.randn(tlen,3) )
    flat_tracks, breaks = get_tracks.flatten_tracks(tracks)
    yield assert_true, len(breaks) == len(tracks)
    total_pts = np.array([ len(line) for line in tracks ]).sum()
    yield assert_true, total_pts==breaks.sum()

def test_roundtrip():
    ntracks = 100
    tracks = []
    for n in xrange(ntracks):
        tlen = np.random.randint(0,high=15)
        tracks.append( np.random.randn(tlen,3) )
    flat_tracks, breaks = get_tracks.flatten_tracks(tracks)
    tracks_r = get_tracks.recombine_flattened_tracks(flat_tracks, breaks)
    yield assert_true, all( [(tr==t).all()
                             for tr, t in zip(tracks_r, tracks)] )
