import numpy as np
import nose.tools as nt

from xipy.external import decotest

from xipy.volume_utils import *

def test_is_spatially_aligned():
    r = np.diag(np.ones(4)).astype('d')
    axes = np.arange(3)
    np.random.shuffle(axes)
    r[:3,:3] = np.take( np.diag(np.random.rand(3)), axes, axis=0 )
    aligned_aff = ni_api.AffineTransform.from_params(
        'ijk', 'xyz', r
        )

    yield nt.assert_true, is_spatially_aligned(aligned_aff)
    
    r[:3,:3] = np.random.randn(3,3)
    unaligned_aff = ni_api.AffineTransform.from_params(
        'ijk', 'xyz', r
        )
    yield nt.assert_false, is_spatially_aligned(unaligned_aff)

def test_spatial_axes_lookup():
    r = np.diag(np.ones(4)).astype('d')
    axes = np.arange(3)
    np.random.shuffle(axes)
    r[:3,:3] = np.take( np.diag(np.random.randn(3)), axes, axis=0 )
    aligned_aff = ni_api.AffineTransform.from_params(
        'ijk', 'xyz', r
        )

    mapping = spatial_axes_lookup(aligned_aff)
    axes = axes.tolist()
    yield (nt.assert_equal,
           axes,
           [mapping[k] for k in (0,1,2)])
    yield (nt.assert_equal,
           axes,
           [mapping[k] for k in ('x', 'y', 'z')])
    yield (nt.assert_equal,
           axes,
           [mapping[k] for k in ('SAG', 'COR', 'AXI')])
           
