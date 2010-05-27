try:
    import vtk       
except ImportError:
    raise ImportError('VTK is not installed.')
    
try:
    import numpy as np
except ImportError:
    raise ImportError('Numpy is not installed.')

from enthought.tvtk.api import tvtk

def _lookup(colors):
    ''' Internal function
    Creates a lookup table with given colors.
    
    Parameters
    ------------
    colors : array, shape (N,3)
            Colormap where every triplet is encoding red, green and blue e.g. 
            r1,g1,b1
            r2,g2,b2
            ...
            rN,gN,bN        
            
            where
            0=<r<=1,
            0=<g<=1,
            0=<b<=1,
    
    Returns
    ----------
    vtkLookupTable
    
    '''
        
    colors=np.asarray(colors,dtype=np.float32)
    
    if colors.ndim>2:
        raise ValueError('Incorrect shape of array in colors')
    
    if colors.ndim==1:
        N=1
        
    if colors.ndim==2:
        
        N=colors.shape[0]    
    
    
    lut=vtk.vtkLookupTable()
    lut.SetNumberOfColors(N)
    lut.Build()
    
    if colors.ndim==2:
        scalar=0
        for (r,g,b) in colors:
            
            lut.SetTableValue(scalar,r,g,b,1.0)
            scalar+=1
    if colors.ndim==1:
        
        lut.SetTableValue(0,colors[0],colors[1],colors[2],1.0)

    return lut

def tvtk_line(lines, colors, opacity=1.0):
    ''' Create an actor for one or more lines.    
    
    Parameters
    ----------
    lines :  list of arrays representing lines as 3d points  for example            
            lines=[np.random.rand(10,3),np.random.rand(20,3)]   
            represents 2 lines the first with 10 points and the second with
            20 points in x,y,z coordinates.
    colors : array, shape (N,3)
            Colormap where every triplet is encoding red, green and blue e.g. 
            r1,g1,b1
            r2,g2,b2
            ...
            rN,gN,bN        
            
            where
            0=<r<=1,
            0=<g<=1,
            0=<b<=1
            
    opacity : float, default 1
                    0<=transparency <=1
    linewidth : float, default is 1
                    line thickness
                    
    
    Returns
    ----------
    vtkActor object
    
    Examples
    --------    
    >>> from <???> import fos
    >>> from enthought.mayavi import mlab
    >>> lines=[np.random.rand(10,3),np.random.rand(20,3)]    
    >>> colors=np.random.rand(2,3)
    >>> c=fos.line(lines,colors)
    >>> f = mlab.figure()
    >>> f.scene.add_actor(c)
    '''
    cell_lines = tvtk.CellArray()
    line_scalars = tvtk.FloatArray()
    pts = tvtk.Points()

    LUT = tvtk.to_tvtk(_lookup(colors))
    LUT.table_range = (0, LUT.number_of_colors)
    multicolor = (LUT.number_of_colors > 1)
    list_of_colors = []
    color_level = 0

    lcnts = np.array([ len(line) for line in lines ])
    total_pts = lcnts.sum()
    flattened_pts = np.empty((total_pts, 3), lines[0].dtype)
    lc = 0
    for n in xrange(len(lines)):
        flattened_pts[lc:lc+lcnts[n]] = lines[n]
        cell_lines.insert_next_cell( range(lc, lc+lcnts[n]) )
        if multicolor:
            list_of_colors += [color_level] * lcnts[n]
            color_level += 1
        lc += lcnts[n]

    pts.data.from_array(flattened_pts)
    
    if multicolor:
        line_scalars.from_array( np.array(list_of_colors) )
    else:
        line_scalars.from_array( np.zeros(pts.number_of_points) )

    pd = tvtk.PolyData()
    pd.lines = cell_lines
    pd.points = pts
    pd.point_data.scalars = line_scalars
    pd.point_data.scalars.name = 'lines'
    return pd, LUT

##     mapper = tvtk.PolyDataMapper()
##     mapper.input = pd
##     mapper.color_mode = 'map_scalars'
##     mapper.lookup_table = LUT
##     mapper.scalar_range = ( 0, color_level )

##     actor = tvtk.Actor()
##     actor.mapper = mapper
##     actor.property.opacity = opacity

##     return actor
