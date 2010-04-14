import numpy as np


def list_indices(L, value, start=0):
    
    ''' find the `indices` with specific `value` in list L

    This is implemented as a generator function

    Example:
    --------

    >>> from fos.core.utils import list_indices as lind
    >>> test=[True, False, True, False, False]
    >>> for i in lind(test,False): print i
    1
    3
    4

    '''
    
    try:

        while start<len(L):

            i = L.index(value, start)

            start=i+1

            yield i

    except ValueError:

        pass



def load_vtk_polydata(self,fname):

    ''' Read a vtk polydata file. Mayavi can do this as well but it
    generates some kind of weird error. Because the vtk format is very
    easy to implement we have implemented here a very simple polydata
    reader. Polydata are usually used for surface representations.

    Parameters:
    -----------

    fname: string, filename    


    Returns:
    --------

    vertices: array,shape (N,3)

    polygons: array, shape(M,K), contains the indices of the vertices
    that make polygons. One polygon for every row. E.g. if the polygons
    are triangles then K=3


    '''

    #this needs a check if file handler is used too and also a check if
    #the file exists.
    f=open(self.fname,'r')
        
    lines=f.readlines()

    taglines=[l.startswith('POINTS') or l.startswith('POLYGONS')  for l in lines]

    pts_polys_tags=[i for i in list_indices(taglines,True)]

    if len(pts_polys_tags)<2:

        NameError('This must be the wrong file no polydata in.')

    #read points
            
    pts_index = pts_polys_tags[0]
              
    pts_tag = lines[pts_index].split()

    pts_no = int(pts_tag[1])

    pts=lines[pts_index+1:pts_index+pts_no+1]

    pts=np.array([np.array(p.split(),dtype=np.float32) for p in pts])

    #read polygons
        
    polys_index = pts_polys_tags[1]

    #print polys_index

    polys_tag = lines[polys_index].split()

    polys_no = int(polys_tag[1])

    polys=lines[polys_index+1:polys_index+polys_no+1]

    polys=np.array([np.array(pl.split(),dtype=np.int) for pl in polys])[:,1:]

    return pts,polys


