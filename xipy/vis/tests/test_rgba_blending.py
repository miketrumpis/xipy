impor xipy.vis.rgba_blending as blending

def simple_test():
    arr1 = np.random.randint(0,high=255, size=(30,30,30)).astype('B')

def resample_test():
    #### Some synthetic data
    fx = 30.0; fy = 74.0; fz = 20.0
    sw_3d = np.sin(2*np.pi*( np.arange(30)[:,None,None]/fx + \
                             np.arange(30)[None,:,None]/fy + \
                             np.arange(30)[None,None,:]/fz ))
##     win = np.zeros(30)
##     win[32:96] = np.hanning(15)
##     win_3d = np.power(win[:,None,None] * win[None,:,None] * win[None,None,:], 1/3.)
##     sw_3d *= win_3d

    main_dr = np.array([1.0]*3)
    main_r0 = np.array([-15.]*3)
    
    rn_3d = np.random.randn(17,23,14)
    over_dr = (np.array(sw_3d.shape,'i')/np.array(rn_3d.shape,'i')).astype('d')
    over_r0 = -(over_dr*rn_3d.shape)/2

    #### Colormap the scalar data
    main_bytes1 = rgba_blending.normalize_and_map(sw_3d, cm.gray)
    main_bytes2 = main_bytes1.copy()
    
    over_bytes1 = rgba_blending.normalize_and_map(rn_3d, cm.hot, alpha=.25)
    # also with an alpha function, rather than a constant alpha--
    # emphasizes larger + and - numbers
    mn = rn_3d.min(); mx = rn_3d.max()
    lx = np.linspace(mn, mx, 256)
    lx *= 2*np.pi/max(abs(mn), abs(mx))
    afunc = np.abs(np.arctan(lx)) * (255*2/np.pi)
    over_bytes2 = rgba_blending.normalize_and_map(rn_3d, cm.jet, alpha=afunc)
    
    rgba_blending.resample_and_blend(main_bytes1, main_dr, main_r0,
                                     over_bytes1, over_dr, over_r0)
    rgba_blending.resample_and_blend(main_bytes2, main_dr, main_r0,
                                     over_bytes2, over_dr, over_r0)

