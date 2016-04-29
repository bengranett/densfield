"""spherical geometry utilities"""


import numpy as N

# degrees to radian conversions
c = N.pi/180
ic = 180/N.pi

def lonlat2xyz(lon,lat,r=1):
    """ """
    x = r*N.cos(lon*c)*N.cos(lat*c)
    y = r*N.sin(lon*c)*N.cos(lat*c)
    z = r*N.sin(lat*c)
    return x,y,z

def xyz2lonlat(x,y,z,getr=True):
    """ """
    if getr:
        r = N.sqrt(x*x+y*y+z*z)
    else:
        r = N.ones(x.shape)
    lat = N.arcsin(z/r)*ic
    lon = N.arctan2(y,x)*ic
    if getr:
        return lon,lat,r
    return lon,lat

def rotate_xyz(x,y,z,angles=None,inverse=False):
    """ Rotate a set of vectors pointing in the direction x,y,z

    angles is a list of longitude and latitude angles to rotate by.
    First the longitude rotation is applied (about z axis), then the
    latitude angle (about y axis).
    """
    if angles==None:
        return x,y,z

    xyz = N.array([x,y,z])
    m = N.eye(3)
    for dlon,dlat in angles:
        dlon*=c
        dlat*=c
        m1 = N.array([[N.cos(dlon),-N.sin(dlon),0],
                      [N.sin(dlon), N.cos(dlon),0],
                      [0,0,1]])

        m2 = N.array([[N.cos(dlat),0,-N.sin(dlat)],
                      [0,1,0],
                      [N.sin(dlat), 0, N.cos(dlat)]])

        m = N.dot(N.dot(m2,m1),m)

    if inverse:
        m = N.linalg.inv(m)
        
    xyz2 = N.dot(m,xyz)
    return xyz2

def rotate_lonlat(lon,lat,angles=[(0,0)]):
    """ Rotate a set of longitude and latitude coordinate pairs.
    """
    xyz = N.array(lonlat2xyz(lon,lat))
    xyz2 = rotate_xyz(*xyz,angles=angles)
    return xyz2lonlat(*xyz2,getr=False)


def test_rotate_lonlat():
    lon = N.array([0,0,0,0])
    lat = N.array([0,30,45,90,])

    
    a,b = rotate_lonlat(lon,lat,angles=[(10,0)])
    assert(N.allclose(b,lat))
    assert(N.allclose(a,10))

    a,b = rotate_lonlat(lon,lat,angles=[(0,-10)])
    assert(N.allclose(b,lat-10))
    assert(N.allclose(a,0))

    lon = N.array([10,10,10,10])
    lat = N.array([0,30,45,90,])


    a,b = rotate_lonlat(lon,lat,angles=[(-10,-10),(10,0)])
    assert(N.allclose(b,lat-10))
    assert(N.allclose(a,lon))


if __name__=="__main__":
    pass
