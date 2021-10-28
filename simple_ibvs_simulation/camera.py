from scipy.spatial.transform import Rotation as R
import cv2 as cv

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
    def __init__(self, translation=(0,0,0), euler=(0, 0, 0), controller="IBVS1", lamb=0.01, noiseStd=0):
        assert controller in ("IBVS1", "IBVS2", "PBVS1", "PBVS2", "PBVS3")

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
        self.controller = controller
        self.lamb = lamb
        self.noiseStd = noiseStd # noise of the projected features

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
        w = vel[3:]
        #w = np.matmul(self.rotation, vel[3:])
        
        self.translation[0] += v[0]
        self.translation[1] += v[1]
        self.translation[2] += v[2]

        r = R.from_matrix(self.rotation)
        rw = R.from_rotvec(w)
        #r = rw*r
        r = r*rw
        self.rotation = r.as_matrix()

    def interactionMatrix(self, X, y, z):
        return [[y/X, -1/X, 0, z, y*z, -(y*y+1)],
                [z/X, 0, -1/X, -y, 1+z*z, -z*y]]

    def _interactionMatrix(self, x, y, Z):
        return [[-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                [0, -1/Z, y/Z, 1+y*y, -x*y, -x]]

    def control(self, targets, features):
        lamb = self.lamb
        noiseStd = self.noiseStd

        if self.controller in ("IBVS1", "IBVS2", "IBVS3"):
            return self._controlIBVS(targets, features, lamb, noiseStd)
        elif self.controller in ("PBVS1", "PBVS2"):
            return self._controlPBVS(targets, features, lamb, noiseStd)
        else:
            raise Exception("Invalid controller '{}'".format(self.controller))

    def _controlPBVS(self, targets, featureSet, lamb, noiseStd):
        """
        targetTranlation and targetRotation expressed in feature frame
        """
        features = np.array(featureSet.features)
        # hard coded
        targetTranslation = np.array([0, -3.33, 0])
        #targetTranslation = np.array([3, 0, 0]) # ???
        targetRotation = R.from_euler("XYZ", (0, 0, np.pi/2)).as_matrix()

        projectedFeatures = np.array([self.globalToImage(f) for f in featureSet.transformedFeatures()])
        projectedFeatures += np.random.normal(0, noiseStd, projectedFeatures.shape)
        
        # convert to correct image coordinates for solvePnP
        projectedFeatures = np.array([(-y, -z) for y, z in projectedFeatures])
        features = np.array(features, dtype=np.float32)
        success, rotation, translation = cv.solvePnP(features, 
                                                     projectedFeatures, 
                                                     #np.array([[self.f, 0, self.imSize/2], [0, self.f, self.imSize/2], [0, 0, 1]]), 
                                                     np.array([[self.f, 0, 0], [0, self.f, 0], [0, 0, 1]]), 
                                                     distCoeffs=np.zeros((4,1), dtype=np.float32), 
                                                     useExtrinsicGuess=False,
                                                     tvec=None,
                                                     rvec=None,
                                                     flags=cv.SOLVEPNP_ITERATIVE)

        translation = translation[:, 0] # feature/object translation wrt camera
        rotation = rotation[:, 0]       # feature/object rotation wrt camera
        
        rotation = R.from_rotvec(rotation).as_matrix()
        # to account for that this camera has x-axis pointing forward, y-axis left and z-axis up
        rotDelta = R.from_euler("XYZ", (-np.pi/2, np.pi/2, 0)).as_matrix()
        rotation = np.matmul(rotDelta, rotation)
        translation = np.matmul(rotDelta, translation)        
        print("Range:", np.linalg.norm(translation))

        translationObjectWRTTarget = -np.matmul(targetRotation.transpose(), targetTranslation)
        translationCamWRTTarget = translationObjectWRTTarget - np.matmul(np.matmul(targetRotation.transpose(), rotation.transpose()), translation)

        rotationCamWRTTarget = np.matmul(targetRotation.transpose(), rotation.transpose())
        rotationCamWRTTargetRotVec = R.from_matrix(rotationCamWRTTarget).as_rotvec()

        
        if self.controller == "PBVS1":
            # PBVS version 1 in chaumette
            def skew(m):
                return [[   0, -m[2],  m[1]], 
                        [ m[2],    0, -m[0]], 
                        [-m[1], m[0],     0]]

            Lx = [] # TODO
            v = -lamb*(translationObjectWRTTarget-translation + np.matmul(np.linalg.matrix_power(skew(translationCamWRTTarget), 1), rotationCamWRTTargetRotVec))
            w = -lamb*rotationCamWRTTargetRotVec
        
        elif self.controller == "PBVS2":
            # PBVS version 2 in chaumette
            Lx = [] # TODO
            v = -lamb*np.matmul(rotationCamWRTTarget.transpose(), translationCamWRTTarget)
            w = -lamb*rotationCamWRTTargetRotVec

        else:
            raise Exception("Invalid controller '{}'".format(self.controller))


        velocity = np.concatenate((v, w))

        err = np.concatenate((translationCamWRTTarget, rotationCamWRTTargetRotVec))

        return velocity, err


    def _controlIBVS(self, targets, featureSet, lamb, noiseStd):
        """
        TODO: should only take projected features as argument and estimate Z
        """
        features = featureSet.transformedFeatures()
        projectedFeatures = np.array([self.globalToImage(f) for f in features])
        projectedFeatures += np.random.normal(0, noiseStd, projectedFeatures.shape)
        success, rotation, translation = cv.solvePnP(np.array(features), 
                                                     np.array([(-y, -z) for y, z in projectedFeatures]), 
                                                     np.array([[1, 0, 0], [0, 1, 0], [0, 0, self.f]]), 
                                                     distCoeffs=np.zeros((4,1), dtype=np.float32), 
                                                     useExtrinsicGuess=False,
                                                     tvec=None,
                                                     rvec=None,
                                                     flags=cv.SOLVEPNP_ITERATIVE)

        #print(translation)
        estX = translation[2][0] # estimate depth 
        #print(estX)

        Lx = []

        YDesired = np.linalg.norm(np.array(features[1]) - np.array(features[0]))/2       

        for feat, target in zip(projectedFeatures, targets):
            y = feat[0]
            z = feat[1]
            #X = np.dot(np.array(feat) - np.array(self.translation), self.rotation[:, 0])
            #y = np.dot(np.array(feat) - np.array(self.translation), self.rotation[:, 1]) / X
            #z = np.dot(np.array(feat) - np.array(self.translation), self.rotation[:, 2]) / X

            Le = self.interactionMatrix(estX, y, z)

            y = target[0]
            z = target[1]
            
            Y = YDesired*np.sign(y) # hard coded
            LeStar = self.interactionMatrix(Y/y, y, z)

            LeLeStar = (np.array(Le) + np.array(LeStar))/2

            if self.controller == "IBVS1":
                # IBVS version 1 in chaumette
                L = Le
            elif self.controller == "IBVS2":
                # IBVS version 2 in chaumette
                L = LeStar
            elif self.controller == "IBVS3":
                # IBVS version 3 in chaumette
                L = LeLeStar
            else:
                raise Exception("Invalid controller '{}'".format(self.controller))

            #Lx.append(np.row_stack(L))
            Lx.append(L)

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
    


