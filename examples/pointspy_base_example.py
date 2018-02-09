import numpy as np
#import pointspy


if __name__ == '__main__':
    
    data = [
                [
                    (0,4,1),
                    (2,3,4),
                    (5,6,7),
                    (0,0,3),
                    (2,5,2),
                    (2,5,7),
                    (3,3,2),
                    (4,5,6),
                    (7,8,2)
                ],
                [ 1,2,3,4,5,4,3,2,1 ]
        ]
    dtype = [('coords',float,3),('attribute',int)]

    data = np.array(zip(*data),dtype=dtype).view(np.recarray)
    proj = pointspy.projection.Proj()
    
    geoRecords = pointspy.GeoRecords(proj,data)

    print geoRecords
    indexKD = geoRecords.indexKD()
    print indexKD.ball([1,2,3],0.5)
