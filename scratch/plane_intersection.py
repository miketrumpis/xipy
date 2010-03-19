import numpy as np
import warnings

def intersecting_tracks(tracks, p0, normal):
##     warnings.filterwarnings('ignore',
##                             message='Warning: divide by zero encountered in divide')
    p0 = np.asarray(p0)
    normal = np.asarray(normal)
    warnings.simplefilter('ignore')
    p0.shape = (1,3)    
    i_tracks = []

    if np.dot(normal,normal) == 1 and (normal==1).any():
        col = np.argwhere(normal==1)[0][0]
        slicer = (slice(None), slice(col, col+1))
        dot_func = lambda x: x[slicer]
    else:
        dot_func = lambda x: np.dot(x,normal)
    
    for n, line in enumerate(tracks):
        d0 = p0 - line[:-1]
        d1 = line[1:] - line[:-1]
        num = dot_func(d0)
        den = dot_func(d1)
        u = num/den
        if ( (u<=1) & (u>=0) ).any():
            i_tracks.append(n)
        
    p0.shape = (3,)
    warnings.resetwarnings()
    return i_tracks

