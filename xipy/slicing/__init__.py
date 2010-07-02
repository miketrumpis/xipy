import nipy.core.api as ni_api
xipy_ras = ni_api.lps_output_coordnames

SAG, COR, AXI = 0, 1, 2
canonical_axes = dict( zip( ('SAG', 'COR', 'AXI'), (SAG, COR, AXI) ) )


def transverse_plane_lookup(idx):
    if idx==SAG:
        return 1, 2
    elif idx==COR:
        return 0, 2
    elif idx==AXI:
        return 0, 1
    raise ValueError('invalid plane index: %d'%idx)

def enumerated_axes(axes):
    if type(axes)==str:
        axes = [c for c in axes]
    if type(axes) not in (tuple, list):
        raise ValueError('Could not parse axes argument')
    enumerated = []
    ras = xipy_ras
    for ax in axes:
        if type(ax)==str:
            if ax in (ras[0], 'x', 'SAG'):
                enumerated.append(0)
            if ax in (ras[1], 'y', 'COR'):
                enumerated.append(1)
            if ax in (ras[2], 'z', 'AXI'):
                enumerated.append(2)
        if type(ax)==int:
            # MODULO 3? might as well make it safe down the line
            enumerated.append(ax%3)
    return enumerated
    
def load_resampled_slicer(image, bbox=None, fliplr=False, mask=False):
    from xipy.slicing.image_slicers import ResampledVolumeSlicer
    from xipy.io import load_spatial_image
    if type(image) in (str, unicode):
        return ResampledVolumeSlicer(load_spatial_image(image),
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
    from xipy.io import load_spatial_image
    if type(overlay) in (str, unicode):
        return SampledVolumeSlicer(load_spatial_image(overlay), bbox=bbox,
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
