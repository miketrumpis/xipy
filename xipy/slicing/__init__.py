import nipy.core.api as ni_api

SAG, COR, AXI = 0, 1, 2
canonical_axes = dict( zip( ('SAG', 'COR', 'AXI'), (SAG, COR, AXI) ) )

def transverse_plane_lookup(idx):
    if idx==SAG:
        return 1, 2
    elif idx==COR:
        return 0, 2
    elif idx==AXI:
        return 0, 1
    raise ValueError('invalid plane index: %d, %s'%idx, canonical_axes[idx])

def _load_image(image):
    from nipy.io.api import load_image as ni_load
    img = ni_load(image)
    return img

def load_resampled_slicer(image, bbox=None, fliplr=False, mask=False):
    from xipy.slicing.image_slicers import ResampledVolumeSlicer
    if type(image) in (str, unicode):
        return ResampledVolumeSlicer(_load_image(image),
                                     bbox=bbox,
                                     mask=mask,
                                     fliplr=fliplr)
    elif type(image) is ni_api.Image:
        return ResampledVolumeSlicer(image,
                                     bbox=bbox,
                                     mask=mask,
                                     fliplr=fliplr)
    elif type(image) is ResampledVolumeSlicer:
        return image
    else:
        raise ValueError('unknown type for image argument '+str(type(image)))

def load_sampled_slicer(overlay, bbox=None, grid_spacing=None,
                        mask=False, fliplr=False):
    from xipy.slicing.image_slicers import SampledVolumeSlicer
    if type(overlay) in (str, unicode):
        return SampledVolumeSlicer(_load_image(overlay), bbox=bbox,
                                   mask=mask,
                                   grid_spacing=grid_spacing,
                                   fliplr=fliplr)
    elif type(overlay) is ni_api.Image:
        return SampledVolumeSlicer(overlay, bbox=bbox,
                                   mask=mask,
                                   grid_spacing=grid_spacing,
                                   fliplr=fliplr)
    elif type(overlay) is SampledVolumeSlicer:
        return overlay
    else:
        raise ValueError('unknown type for image argument '+str(type(overlay)))
