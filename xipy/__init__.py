__docformat__ = 'restructuredtext'
import os
from enthought.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'
TEMPLATE_MRI_PATH = os.path.join(os.path.dirname(__file__),
                                 'resources/template_T1_1mm_brain.nii.gz')
