from scipy.spatial.transform import Rotation as R
import numpy as np

from coordinate_system import CoordinateSystem, CoordinateSystemArtist

class Camera(CoordinateSystem):
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, cameraMatrix, *args, **kwargs):
        CoordinateSystem.__init__(self, *args, **kwargs)
        
        self.cameraMatrix = cameraMatrix
        
        self.detectedFeaturePoints = []
        self.projectedFeaturePoints = []

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

    def detectFeatures(self, featurePoints3D):
        #featurePoints3D = np.array(featurePoints3D) - np.array(self.translation)
        #featurePoints3D = np.matmul(np.linalg.inv(np.array(self.rotation)), np.array(featurePoints3D).transpose()).transpose()
        #self.detectedFeaturePoints = list(featurePoints3D)
        if not featurePoints3D:
            self.detectedFeaturePoints = []
            self.projectedFeaturePoints = []
            return
        featurePoints3D = self.transformedPointsInv(featurePoints3D)
        self.detectedFeaturePoints = featurePoints3D
        self.projectedFeaturePoints = []
        for feature in featurePoints3D:
            detectedPoint = self.projectLocal3D(feature)
            self.projectedFeaturePoints.append(detectedPoint)

    def project(self, point):
        """
        TODO: Handle singular matrix
        """
        l0 = np.array(point) # position vector for line
        vl = l0 - np.array(self.translation) # direction vector for line
        p0 = np.array(self.translation) + np.array(self.rotation)[:, 2]*self.f # center of plane
        mat = np.column_stack((np.array(self.rotation)[:, 0], np.array(self.rotation)[:, 1], -vl))
        x = np.linalg.solve(mat, l0-p0)

        # Check if points are within image plane
        if (abs(x[0]) > self.imSize/2 or abs(x[1]) > self.imSize/2):
            #print("Projection: Point not within image plane")
            return False

        # Check if the projection is from the "front" of the camera/plane
        if np.dot(np.array(self.rotation)[:, 2], vl) < 0:
            #print("Projection: Point behind image plane")
            return False

        return l0 + x[2]*vl

    def projectLocal3D(self, point):
        """
        TODO: Handle singular matrix
        """
        l0 = np.array(point) # position vector for line
        vl = l0
        p0 = np.array([0, 0, self.f]) # center of plane
        mat = np.column_stack((np.array([1, 0, 0]).transpose(), np.array([0, 1, 0]).transpose(), -vl))
        x = np.linalg.solve(mat, l0-p0)

        # Check if points are within image plane
        if (abs(x[0]) > self.imSize/2 or abs(x[1]) > self.imSize/2):
            #print("Projection: Point not within image plane")
            return False

        # Check if the projection is from the "front" of the camera/plane
        if np.dot(np.array([0, 0, 1]), vl) < 0:
            #print("Projection: Point behind image plane")
            return False

        return l0 + x[2]*vl

    """
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
    """

class CameraArtist(CoordinateSystemArtist):
    def __init__(self, camera, nFeaturesToBeDetected, *args, **kwargs):
        CoordinateSystemArtist.__init__(self, camera, *args, **kwargs)

        self.camera = camera
        self.nFeaturesToBeDetected = nFeaturesToBeDetected # remove and fix this in pose estimation object

        self.imagePoints = None
        self.focalLine = None
        self.topLine = None

        self.projectedFeaturePoints = None
        self.detectedFeaturePoints = None
        self.referenceLines = []
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
                self.imagePoints,
                self.projectedFeaturePoints,
                self.detectedFeaturePoints,
                *self.referenceLines] + CoordinateSystemArtist.artists(self)

    def init(self, ax):
        CoordinateSystemArtist.init(self, ax)
        self.focalLine = ax.plot3D([], [], [], color="black")[0]
        self.topLine = ax.plot3D([], [], [], color="black")[0]
        self.imagePoints = ax.plot3D([], [], [], color="r")[0]
        self.projectedFeaturePoints = ax.plot3D([], [], [], color="cornflowerblue", marker=".")[0] # plot3D
        self.detectedFeaturePoints = ax.plot3D([], [], [], color="red", marker="o")[0] # plot3D
        #self.projectedFeaturePoints = ax.scatter([], [], [], color="cornflowerblue", marker=".") # scattering
        self.referenceLines = []
        for _ in range(self.nFeaturesToBeDetected): # hard coded
            self.referenceLines.append(ax.plot3D([], [], [], color="grey")[0])

        return self.artists()

    def update(self, showAxis=True, referenceTranslation=(0,0,0)):
        CoordinateSystemArtist.update(self, showAxis, referenceTranslation)
        points = self.camera.transformedPoints(self.camera.imagePoints, referenceTranslation)
        topLine = self.camera.transformedPoints(self.camera.topLine, referenceTranslation)
        focalLine = self.camera.transformedPoints(self.camera.focalLine, referenceTranslation)

        self.focalLine.set_data_3d(*zip(*focalLine))
        self.topLine.set_data_3d(*zip(*topLine))
        self.imagePoints.set_data_3d(*zip(*points + [points[0]]))

        #projPoints = list(filter(lambda p: p is not False, self.camera.projectedFeaturePoints))
        projPoints = []
        featurePoints3D = []
        for pp, f in zip(self.camera.projectedFeaturePoints, self.camera.detectedFeaturePoints):
            if pp is not False:
                projPoints.append(pp)
                featurePoints3D.append(f)

        if projPoints:
            projPoints = self.camera.transformedPoints(projPoints, referenceTranslation) # local to global
            self.projectedFeaturePoints.set_data_3d(*zip(*projPoints + [projPoints[0]]))
            featurePoints3D = self.camera.transformedPoints(featurePoints3D, referenceTranslation) # local to global
            self.detectedFeaturePoints.set_data_3d(*zip(*featurePoints3D + [featurePoints3D[0]]))
        else:
            self.projectedFeaturePoints.set_data_3d([], [], [])
            self.detectedFeaturePoints.set_data_3d([], [], [])   
        
        if not self.camera.detectedFeaturePoints:
            for l in self.referenceLines:
                l.set_data_3d([], [], [])
        for l, pp, f in zip(self.referenceLines, self.camera.projectedFeaturePoints, self.camera.detectedFeaturePoints):
            if pp is not False:
                pp = self.camera.transformedPoints([pp], referenceTranslation)[0]
                f = self.camera.transformedPoints([f], referenceTranslation)[0]
                l.set_data_3d(*zip(pp, f))
            else:
                l.set_data_3d([], [], [])

        """
        # Scattering
        
        if projPoints:
            projPoints = np.array(projPoints) - np.array(referenceTranslation)
            self.projectedFeaturePoints._offsets3d = [*zip(*projPoints + [projPoints[0]])]
        else:
            self.projectedFeaturePoints._offsets3d = ((), (), ())
        """     

        return self.artists()

if __name__ == "__main__":
    pass
    


