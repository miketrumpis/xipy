import numpy as np
import nose.tools as nt

from xipy.external import decotest

from xipy.utils import *

@decotest.ipdoctest
def test_voxel_index_list():
    """
>>> voxel_index_list((2,3), order='ijk')
array([[0, 0],
       [1, 0],
       [0, 1],
       [1, 1],
       [0, 2],
       [1, 2]])
>>> voxel_index_list((2,3), order='kji')
array([[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2]])
    """

@decotest.ipdoctest
def test_coord_list_to_mgrid():
    """
>>> coords_list = voxel_index_list((2,3), order='ijk')
>>> mgrid = coord_list_to_mgrid(coords_list, (2,3), order='ijk')
>>> mgrid
array([[[0, 0, 0],
        [1, 1, 1]],
<BLANKLINE>
       [[0, 1, 2],
        [0, 1, 2]]])
>>> coords_list = voxel_index_list((2,3), order='kji')
>>> mgrid = coord_list_to_mgrid(coords_list, (2,3), order='kji')
>>> mgrid
array([[[0, 0, 0],
        [1, 1, 1]],
<BLANKLINE>
       [[0, 1, 2],
        [0, 1, 2]]])

    """

def test_bad_voxel_type():
    voxels = np.random.randn(10,3) + 1j*np.random.randn(10,3)
    location = np.random.randn(3)
    yield nt.assert_raises, ValueError, closest_voxel, voxels, location

    voxels = voxels.real
    location = np.random.randn(4)
    yield nt.assert_raises, ValueError, closest_voxel, voxels, location

def test_voxel_lookup():
    voxels = voxel_index_list((3,3))
    location = [1,0] # would be the 2nd entry, in 'ijk' order
    idx, dist = closest_voxel(voxels, location)
    yield nt.assert_true, idx==1, 'expected idx not matched'
    yield nt.assert_true, dist==0, 'expected distance not matched'

    location = [-1, 0]
    idx, dist = closest_voxel(voxels, location)
    yield nt.assert_true, dist==1, 'expected distance not matched'

def test_voxel_lookup_by_ref():
    def reference_lookup(vx, loc):
        idx = np.argmin( ( (vx-loc)**2 ).sum(axis=1) )
        dist = ( (vx[idx] - loc)**2 ).sum()**0.5
        return idx, dist

    vox = np.random.randn(10,3)
    loc = np.random.randn(3)
    i1, d1 = closest_voxel(vox, loc)
    i2, d2 = reference_lookup(vox, loc)
    yield nt.assert_true, i1==i2, 'closest voxel does not match reference'
    yield nt.assert_true, abs(d2-d1) < 1e-10, 'closest voxel does not match reference'

def test_mni_db():
    for db_type in ('icbm', 'brett'):
        db = MNI_to_Talairach_db(db=db_type)

        loc = db.locations[0]
        labels = [db.table[i][db.lut[0]] for i in xrange(len(db.table))]

        retrieved_labels = db(loc)

        yield (
            nt.assert_true,
            all([x==y for x,y in zip(labels, retrieved_labels)])
            )
