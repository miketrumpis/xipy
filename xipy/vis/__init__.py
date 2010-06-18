# affects some mpl based classes
BLITTING=True


def quick_ortho_plot(x,y,z,bbox=[]):
    import matplotlib.pyplot as pp
    import xipy.volume_utils as vu
    from xipy.vis.single_slice_plot import SliceFigure
    # find or make the x, y, z plane extents
    if bbox:
        extents = vu.limits_to_extents(bbox)
    else:
        extents = [ reduce(lambda x,y: x+y, zip([0,0],p.shape[:2][::-1]))
                    for p in (x, y, z) ]
        
    
    fig = pp.figure()

    ax_z = fig.add_subplot(131)
    sf_z = SliceFigure(fig, extents[2])
    img_z = sf_z.spawn_image(z, extent=extents[2])
    ax_z.set_title('plot z')

    ax_y = fig.add_subplot(132)
    sf_y = SliceFigure(fig, extents[1])
    img_y = sf_y.spawn_image(y, extent=extents[1])
    ax_y.set_title('plot y')
    
    ax_x = fig.add_subplot(133)
    sf_x = SliceFigure(fig, extents[0])
    img_x = sf_x.spawn_image(x, extent=extents[0])
    ax_x.set_title('plot x')

    pp.show()
    return fig

    
