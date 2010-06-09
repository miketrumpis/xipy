import numpy as np
import numpy.testing as npt
from nose.tools import assert_true, assert_equal, assert_false

import nipy.core.api as ni_api

from xipy.external import decotest
import xipy.vis.color_mapping as cm

# the code to test
from xipy.vis.rgba_blending import *


def gen_img(shape=(10,20,12)):
    scalars = np.random.randn(*shape)
    return ni_api.Image(scalars,
                        ni_api.AffineTransform.from_params(
                            'ijk', ni_api.ras_output_coordnames, np.eye(4)
                            )
                        )

    
