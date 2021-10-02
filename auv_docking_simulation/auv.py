from scipy.spatial.transform import Rotation as R
import numpy as np

from coordinate_system import CoordinateSystem, CoordinateSystemArtist

class AUV(CoordinateSystem):
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, camera, *args, **kwargs):
        CoordinateSystem.__init__(self, *args, **kwargs)

        self.camera = camera

        # Points
        self.l = 3.0
        self.w = 10
        self.bodyPoints = [[-self.l/2, 0, 0], [self.l/2, 0, 0]]
        
        self.relativeCameraRotation = R.from_euler("XYZ", (-np.pi/2, -np.pi/2, 0)).as_matrix()
        self.setCameraTransform()

    def setCameraTransform(self):
        cameraTranslation = np.array(self.translation) + np.matmul(self.rotation, np.array([-self.l/2, 0, 0]))
        cameraRotation = np.matmul(self.rotation, self.relativeCameraRotation)
        self.camera.setTransform(cameraTranslation, cameraRotation)

    def move(self, vel):
        v = np.matmul(self.rotation, vel[:3])
        w = np.matmul(self.rotation, vel[3:])
        
        self.translation[0] += v[0]
        self.translation[1] += v[1]
        self.translation[2] += v[2]

        r = R.from_matrix(self.rotation)
        rw = R.from_rotvec(w)
        r = rw*r
        self.rotation = list(r.as_matrix())
        self.setCameraTransform()


class AUVArtist(CoordinateSystemArtist):
    def __init__(self, auv, *args, **kwargs):
        CoordinateSystemArtist.__init__(self, auv, *args, **kwargs)
        
        self.auv = auv
        self.bodyPoints = None

    def artists(self):
        return [self.bodyPoints,
                self.xAxis,
                self.yAxis,
                self.zAxis] + CoordinateSystemArtist.artists(self)

    def init(self, ax):
        CoordinateSystemArtist.init(self, ax)

        self.bodyPoints = ax.plot3D([], [], [], color="y", linewidth=self.auv.w)[0]
        return self.artists()

    def update(self):
        CoordinateSystemArtist.update(self)

        bodyPoints = self.auv.transformedPoints(self.auv.bodyPoints)
        self.bodyPoints.set_data_3d(*zip(*bodyPoints))

        return self.artists()

if __name__ == "__main__":
    pass
    


