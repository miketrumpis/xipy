from nipy.io.api import load_image as ni_load_image
import nipy.core.api as ni_api
import numpy as np

def load_image(filename):
    input_coords = dict( zip(range(3), 'ijk') )
    output_coords = dict( zip(ni_api.lps_output_coordnames,
                              ni_api.ras_output_coordnames) )
    img = ni_load_image(filename)
    cm = img.coordmap.renamed_domain(input_coords)
    cm = cm.renamed_range(output_coords)
    
    return ni_api.Image(np.asarray(img), cm)
    
