from scipy.spatial.transform import Rotation as R
import numpy as np

class CoordinateSystem:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, translation=(0,0,0), euler=(0, 0, 0)):
        r = R.from_euler("XYZ", euler)
        self.rotation = r.as_matrix()
        self.initialRotation = r.as_matrix()
        self.translation = list(translation)
        self.initialTranslation = list(translation)

    def reset(self):
        self.translation = self.initialTranslation.copy()
        self.rotation = self.initialRotation

    def setTransform(self, trans, rot):
        self.translation = [_ for _ in trans]
        self.rotation = [_ for _ in rot]

    def transformedPoints(self, points):
        transformedPoints = []
        for p in points:
            transformedPoints.append(np.matmul(self.rotation, np.array(p)))

        newPoints = []
        for p in transformedPoints:
            newPoints.append( [p1+p2 for p1,p2 in zip(p, self.translation)] )

        return newPoints

class CoordinateSystemArtist:
    def __init__(self, coordinateSystem):
        self.cs = coordinateSystem

        self.xAxis = None
        self.yAxis = None
        self.zAxis = None

    def artists(self):
        return [self.xAxis,
                self.yAxis,
                self.zAxis]

    def init(self, ax):
        self.xAxis = ax.plot3D([], [], [], color="r", linewidth=2)[0]
        self.yAxis = ax.plot3D([], [], [], color="g", linewidth=2)[0]
        self.zAxis = ax.plot3D([], [], [], color="b", linewidth=2)[0]

        return self.artists()

    def update(self):
        origin = np.array(self.cs.translation)
        rotation = np.array(self.cs.rotation)
        self.xAxis.set_data_3d(*zip(*[origin, origin + rotation[:, 0]]))
        self.yAxis.set_data_3d(*zip(*[origin, origin + rotation[:, 1]]))
        self.zAxis.set_data_3d(*zip(*[origin, origin + rotation[:, 2]]))

        return self.artists()
