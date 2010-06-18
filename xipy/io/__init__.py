from nipy.io.api import load_image as ni_load_image
import nipy.core.api as ni_api
import numpy as np

from copy import copy

def load_image(filename):
    img = ni_load_image(filename)
    # wtf?
    if np.asarray(img).dtype.char == 'h':
        img = ni_api.Image(np.asarray(img).astype('h'), img.coordmap)
    input_coords = dict( zip(range(3), 'ijk') )
    output_coords = dict( zip(ni_api.lps_output_coordnames,
                              ni_api.ras_output_coordnames) )
    cm = img.coordmap.renamed_domain(input_coords)
    cm = cm.renamed_range(output_coords)
    
    return ni_api.Image(np.asarray(img), cm)

def load_spatial_image(filename):
    img = load_image(filename)
    in_coords = list(img.coordmap.function_domain.coord_names)
    other_coords = filter(lambda x: x not in 'ijk', in_coords)
    if other_coords:
        idata = np.asanyarray(img)
        cm_3d = copy(img.coordmap)
        while other_coords:
            c = other_coords.pop()
            slicing = [slice(None)] * idata.ndim
            slicing[in_coords.index(c)] = 0
            idata = idata[slicing]
            cm_3d = ni_api.drop_io_dim(cm_3d, c)
        img = ni_api.Image(idata.copy(), cm_3d)
        del idata
    return img
        
    
    
