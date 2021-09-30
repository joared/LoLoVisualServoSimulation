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



class Camera:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, translation=(0,0,0), euler=(0, 0, 0)):
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

    def reset(self):
        self.rotation = self.initialRotation
        self.translation = self.initialTranslation.copy()

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


    
if __name__ == "__main__":
    pass
    


