import xipy.vis.rgba_blending as blending
import xipy.volume_utils as vu
import numpy as np
from nose.tools import assert_true, assert_equal, assert_false
from matplotlib import cm
import nipy.core.api as ni_api

def simple_test():
    arr1 = np.random.randint(0,high=255, size=(10,10,10,4)).astype('B')
    arr2 = np.random.randint(0,high=255, size=(10,10,10,4)).astype('B')
    dr = np.array([1,1,1])
    r0 = np.array([0,0,0])

    # test blend arr2 into arr1 with alpha2 = 0
    a1 = arr1.copy()
    a2 = arr2.copy()
    a2[...,3] = 0
    blending.resample_and_blend(a1, dr, r0, a2, dr, r0)
    yield assert_true, (a1==arr1).all()

    # test blending arr2 into arr1 with alpha2 = 128
    a1 = arr1.copy()
    a2 = arr2.copy()
    a2[...,3] = 128
    blending.resample_and_blend(a1, dr, r0, a2, dr, r0)
    yield assert_true, (a1[...,:3]==((arr1[...,:3].astype('i') + \
                                      arr2[...,:3].astype('i'))/2)).all()

    # test blending arr2 into arr1 with alpha2 = 255
    # hmm.. there is a quirk in the algorithm where the maximal alpha
    # value where arr2 will completely replace arr1 would be 256. BUT
    # these are 8bit ints, so 255 is the max!! Therefore, check to see that
    # the maximum error is +/- 1
    a1 = arr1.copy()
    a2 = arr2.copy()
    a2[...,3] = 255
    blending.resample_and_blend(a1, dr, r0, a2, dr, r0)
    yield assert_true, np.abs(a1[...,:3].astype('i')-\
                              arr2[...,:3].astype('i')).max() == 1


def simple_resample_test():
    blended_block = np.zeros((10,10,10,4), 'B')
    over_block = np.ones((4,4,4,4), 'B')*255
    over_block[:1,:1,:1] = 0
    over_block[-1:,-1:,-1:] = 0
    main_dr = np.ones(3); main_r0 = np.array([-5.]*3)
    over_dr = np.ones(3)*2; over_r0 = np.array([-4.]*3)
    blending.resample_and_blend(blended_block, main_dr, main_r0,
                                over_block, over_dr, over_r0)
    nz_r = blended_block[...,0].nonzero()
    nz_g = blended_block[...,1].nonzero()
    nz_b = blended_block[...,2].nonzero()
    # check that all nonzero components are the same length
    yield assert_true, len(filter(lambda x: (x[0]==x[1]).all(),
                                  zip(nz_r,nz_g)))==3
    yield assert_true, len(filter(lambda x: (x[0]==x[1]).all(),
                                  zip(nz_g,nz_b)))==3

    main_nz_vox = np.array( nz_r ).T
    over_vox_from_main_vox = ni_api.compose(
        ni_api.Affine.from_start_step('ijk', 'xyz', over_r0, over_dr).inverse,
        ni_api.Affine.from_start_step('ijk', 'xyz', main_r0, main_dr)
        )
    over_nz_vox = over_vox_from_main_vox(main_nz_vox).astype('i')
    # check that in fact the nonzero components from the main blended
    # image are the nonzero components from the overlay image..
    # this mask is False at all over_nz_vox
    mask = np.ma.getmask(vu.signal_array_to_masked_vol(
        np.empty(len(over_nz_vox)), over_nz_vox
        ))
    # where the mask is True should be 0
    yield assert_false, over_block[...,0][mask].any()
    # where the mask is False should be != 0
    yield assert_true, over_block[...,0][np.logical_not(mask)].all()


