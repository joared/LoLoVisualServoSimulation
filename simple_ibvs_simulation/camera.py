from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from itertools import product, combinations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

import time

class FeatureSet:
    def __init__(self, features, translation=(0,0,0), euler=(0, 0, 0)):
        self.features = features
        r = R.from_euler("XYZ", euler)
        self.rotation = r.as_matrix()
        self.translation = list(translation)

    def transformedFeatures(self):
        transformedPoints = []
        for p in self.features:
            transformedPoints.append(np.matmul(self.rotation, np.array(p)))

        points = []
        for p in transformedPoints:
            points.append( [p1+p2 for p1,p2 in zip(p, self.translation)] )

        return points

class Camera:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, translation=(0,0,0), euler=(0, 0, 0), controlRule="Le"):
        f = 1 # focal length
        self.f = f
        self.imSize = 2
        self.focalLine = [[0, 0, 0], [f, 0, 0]]
        self.points = [[f, self.imSize/2, -self.imSize/2], 
                       [f, -self.imSize/2, -self.imSize/2], 
                       [f, -self.imSize/2, self.imSize/2], 
                       [f, self.imSize/2, self.imSize/2]]

        r = R.from_euler("XYZ", euler)
        self.rotation = r.as_matrix()
        self.initialRotation = r.as_matrix()
        self.translation = list(translation)
        self.initialTranslation = list(translation)
        self.controlRule = controlRule

    def reset(self):
        self.rotation = self.initialRotation
        self.translation = self.initialTranslation.copy()

    def transformedPoints(self):
        transformedPoints = []
        for p in self.points:
            transformedPoints.append(np.matmul(self.rotation, np.array(p)))

        points = []
        for p in transformedPoints:
            points.append( [p1+p2 for p1,p2 in zip(p, self.translation)] )

        return points

    def transformedFocalLine(self):
        transformedPoints = []
        for p in self.focalLine:
            transformedPoints.append(np.matmul(self.rotation, np.array(p)))

        points = []
        for p in transformedPoints:
            points.append( [p1+p2 for p1,p2 in zip(p, self.translation)] )

        return points

    def transformedTopLine(self):
        start = self.transformedFocalLine()[0]
        end = np.array(start) + self.rotation[:, 0]*self.f + self.rotation[:, 2]*self.imSize/2
        return [start, end]

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

    def move(self, vel):
        """
        Moves camera and adds camera position to trajectory
        """
        v = np.matmul(self.rotation, vel[:3])
        w = np.matmul(self.rotation, vel[3:])
        
        self.translation[0] += v[0]
        self.translation[1] += v[1]
        self.translation[2] += v[2]

        r = R.from_matrix(self.rotation)
        rw = R.from_rotvec(w)
        r = rw*r
        self.rotation = r.as_matrix()

    def interactionMatrix(self, X, y, z):
        return [[y/X, -1/X, 0, z, y*z, -(y*y+1)],
                [z/X, 0, -1/X, -y, 1+z*z, -z*y]]

    def control(self, targets, features, lamb=0.1):
        """
        TODO: should only take projected features as argument and estimate Z
        """

        Lx = []

        YDesired = np.linalg.norm(np.array(features[1]) - np.array(features[0]))/2

        for feat, target in zip(features, targets):
            #Z = np.dot(np.array(feat) - np.array(self.translation), self.rotation[:, 0])
            #x = np.dot(np.array(feat) - np.array(self.translation), -self.rotation[:, 1]) / Z
            #y = np.dot(np.array(feat) - np.array(self.translation), -self.rotation[:, 2]) / Z
            
            X = np.dot(np.array(feat) - np.array(self.translation), self.rotation[:, 0])
            y = np.dot(np.array(feat) - np.array(self.translation), self.rotation[:, 1]) / X
            z = np.dot(np.array(feat) - np.array(self.translation), self.rotation[:, 2]) / X

            Le = self.interactionMatrix(X, y, z)

            y = target[0]
            z = target[1]

            Y = YDesired*np.sign(y) # hard coded
            X = Y/y
            
            LeStar = self.interactionMatrix(X, y, z)

            LeLeStar = (np.array(Le) + np.array(LeStar))/2

            if self.controlRule == "Le":
                L = Le
            elif self.controlRule == "LeStar":
                L = LeStar
            elif self.controlRule == "LeLeStar":
                L = LeLeStar
            else:
                raise Exception("Invalid control rule '{}'".format(self.controlRule))

            Lx.append(np.row_stack(L))

        projectedFeatures = [self.globalToImage(feature) for feature in features]
        Lx = np.row_stack(Lx)
        LxPinv = np.linalg.pinv(Lx)
        err = np.array([v for f in projectedFeatures for v in f]) - np.array([ v for t in targets for v in t])
        v = -lamb*np.matmul(LxPinv, err)

        return v, err

    
if __name__ == "__main__":
    #from animation import CameraAnimator
    from plotter import CameraPLotter
    camera = Camera(translation=(-5, -0.2, -1), euler=(-0.3, 0.1, 0))
    # feature positions (global frame)
    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(1, 0, 1), euler=(0.5, 0, 0))
    # desired position of feature in camera frame (y [left], z [up])
    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]

    #cameraPlotter = CameraPLotter(camera, features, targets)
    cameraAnimator = CameraAnimator(camera, featureSet.transformedFeatures(), targets)
    cameraAnimator.animate()
    cameraAnimator.show()
    exit()
    err = [np.inf]
    # making this threshold to large shows the sensitivity of the pose
    threshold = 0.001
    cameraPlotter.plot()
    while not all([e < threshold for e in err]):
        """
        TODO: plot before controlling
        """
        v, err = camera.control(targets, features, lamb=0.1)
        if v is not False:
            camera.move(v)
        #camera.move((vx, vy, vz, wx, wy, wz))

        cameraPlotter.plot()
        input()

    print("DONE")
    plt.show()
    


