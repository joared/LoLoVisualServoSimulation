from scipy.spatial.transform import Rotation as R
import numpy as np

from coordinate_system import CoordinateSystem, CoordinateSystemArtist

class Camera(CoordinateSystem):
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, cameraMatrix, *args, **kwargs):
        CoordinateSystem.__init__(self, *args, **kwargs)
        
        self.cameraMatrix = cameraMatrix
        
        # Points
        f = 1 # focal length
        self.f = f
        self.imSize = 2

        self.topLine = [[0, 0, 0,], [0, -self.imSize/2, f]]
        self.focalLine = [[0, 0, 0], [0, 0, f]]
        self.imagePoints = [[self.imSize/2, -self.imSize/2, f], 
                       [-self.imSize/2, -self.imSize/2, f], 
                       [-self.imSize/2, self.imSize/2, f], 
                       [self.imSize/2, self.imSize/2, f]]

    def project(self, point):
        """
        TODO: Handle singular matrix
        """
        l0 = np.array(point) # position vector for line
        vl = l0 - np.array(self.translation) # direction vector for line
        p0 = self.transformedFocalLine()[1] # center of plane
        mat = np.column_stack((self.rotation[:, 1], self.rotation[:, 2], -vl))
        x = np.linalg.solve(mat, l0-p0)

        # Check if points are within image plane
        if (abs(x[0]) > self.imSize/2 or abs(x[1]) > self.imSize/2):
            print("Projection: Point not within image plane")
            #return False

        # Check if the projection is from the "front" of the camera/plane
        if np.dot(self.rotation[:, 0], vl) < 0:
            print("Projection: Point behind image plane")
            #return False

        return l0 + x[2]*vl

    def globalToImage(self, point):
        #Z = np.dot(np.array(point) - np.array(self.translation), self.rotation[:, 0])
        #x = np.dot(np.array(point) - np.array(self.translation), -self.rotation[:, 1]) / Z
        #y = np.dot(np.array(point) - np.array(self.translation), -self.rotation[:, 2]) / Z

        # should be called project
        feature = point
        l0 = np.array(feature) # position vector for line
        vl = l0 - np.array(self.translation) # direction vector for line
        p0 = self.transformedFocalLine()[1] # center of plane
        mat = np.column_stack((self.rotation[:, 1], self.rotation[:, 2], -vl))
        x = np.linalg.solve(mat, l0-p0)
        
        # Check if points are within image plane
        if (abs(x[0]) > self.imSize/2 or abs(x[1]) > self.imSize/2):
            print("GlobalToImage: Point not within image plane")
            #return False

        # Check if the projection is from the "front" of the camera/plane
        if np.dot(self.rotation[:, 0], vl) < 0:
            print("GlobalToImage: Point behind image plane")
            #return False

        return (x[0], x[1])

    def imageToGlobal(self, point):
        return self.transformedFocalLine()[1] + self.rotation[:, 1]*point[0] + self.rotation[:, 2]*point[1]

    def rotate(self, r):
        self.rotation = r

class CameraArtist(CoordinateSystemArtist):
    def __init__(self, camera, *args, **kwargs):
        CoordinateSystemArtist.__init__(self, camera, *args, **kwargs)

        self.camera = camera

        self.imagePoints = None
        self.focalLine = None
        self.topLine = None

        """
        # targets on image plane in 2D plot
        self.target2D = None
        self.features3D = None
        self.features2D = None
        self.velocityLines = []
        self.featureTrajectories = []
        """

    def artists(self):
        return [self.focalLine, 
                self.topLine, 
                self.imagePoints] + CoordinateSystemArtist.artists(self)

    def init(self, ax):
        CoordinateSystemArtist.init(self, ax)
        self.focalLine = ax.plot3D([], [], [], color="black")[0]
        self.topLine = ax.plot3D([], [], [], color="black")[0]
        self.imagePoints = ax.plot3D([], [], [], color="r")[0]

        return self.artists()

    def update(self):
        CoordinateSystemArtist.update(self)
        points = self.camera.transformedPoints(self.camera.imagePoints)
        topLine = self.camera.transformedPoints(self.camera.topLine)
        focalLine = self.camera.transformedPoints(self.camera.focalLine)

        self.focalLine.set_data_3d(*zip(*focalLine))
        self.topLine.set_data_3d(*zip(*topLine))
        self.imagePoints.set_data_3d(*zip(*points + [points[0]]))

        return self.artists()

if __name__ == "__main__":
    pass
    


