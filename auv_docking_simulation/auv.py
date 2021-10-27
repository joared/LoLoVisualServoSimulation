from scipy.spatial.transform import Rotation as R
import numpy as np

from coordinate_system import CoordinateSystem, CoordinateSystemArtist

class AUV(CoordinateSystem):
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, 
                 camera,
                 cameraPlacement,
                 size,
                 #relativeCameraTranslation=np.array([-3.0/2, 0, 0]),
                 #relativeCameraRotation=R.from_euler("XYZ", (-np.pi/2, -np.pi/2, 0)).as_matrix(), 
                 *args, 
                 **kwargs):
        CoordinateSystem.__init__(self, *args, **kwargs)

        self.camera = camera

        # Points
        self.l, self.w = size
        self.bodyPoints = [[-self.l/2, 0, 0], [self.l/2, 0, 0]]

        self.trail = []
        
        if cameraPlacement == "back":
            relativeCameraTranslation = np.array([-self.l/2, 0, 0])
            relativeCameraRotation = R.from_euler("XYZ", (-np.pi/2, -np.pi/2, 0)).as_matrix()
        elif cameraPlacement == "front":
            relativeCameraTranslation = np.array([self.l/2, 0, 0])
            relativeCameraRotation = R.from_euler("XYZ", (-np.pi/2, np.pi/2, 0)).as_matrix()
            #relativeCameraRotation = np.eye(3)
        else:
            raise Exception("Invalid camera placement '{}'".format(cameraPlacement))

        self.relativeCameraTranslation = relativeCameraTranslation
        self.relativeCameraRotation = relativeCameraRotation
        self.setCameraTransform()

    def setCameraTransform(self):
        cameraTranslation = np.array(self.translation) + np.matmul(self.rotation, self.relativeCameraTranslation)
        cameraRotation = np.matmul(self.rotation, self.relativeCameraRotation)
        self.camera.setTransform(cameraTranslation, cameraRotation)

    def move(self, vel, dt):
        v = np.matmul(self.rotation, np.array(vel[:3])*dt)
        w = np.array(vel[3:])*dt#np.matmul(self.rotation, np.array(vel[3:])*dt) #why us this not supposed to be multiplied with rotation??
        
        self.translation[0] += v[0]
        self.translation[1] += v[1]
        self.translation[2] += v[2]

        self.trail.append(self.translation.copy())
        if len(self.trail) == 500:
            self.trail = self.trail[-500+20:]

        r = R.from_matrix(self.rotation)
        rw = R.from_rotvec(w)
        #r = rw*r ### which one??
        r = r*rw ### which one??

        def skew(m):
            return [[   0, -m[2],  m[1]], 
                    [ m[2],    0, -m[0]], 
                    [-m[1], m[0],     0]]

        #r = R.from_matrix(np.matmul(skew(w), r.as_matrix()))
        self.rotation = list(r.as_matrix())
        self.setCameraTransform()


class AUVArtist(CoordinateSystemArtist):
    def __init__(self, auv, color="y", *args, **kwargs):
        CoordinateSystemArtist.__init__(self, auv, *args, **kwargs)
        
        self.auv = auv
        self.color = color
        self.bodyPoints = None
        self.trail = None

    def artists(self):
        return [self.bodyPoints,
                self.trail] + CoordinateSystemArtist.artists(self)

    def init(self, ax):
        CoordinateSystemArtist.init(self, ax)
        self.bodyPoints = ax.plot3D([], [], [], color=self.color, linewidth=self.auv.w)[0]
        #self.bodyPoints = ax.plot3D([], [], [], color=self.color)[0]
        self.trail = ax.plot3D([], [], [], color=self.color, marker="^", linewidth=1)[0]

        return self.artists()

    def update(self, showAxis=True, referenceTranslation=(0,0,0)):
        CoordinateSystemArtist.update(self, showAxis, referenceTranslation)

        bodyPoints = self.auv.transformedPoints(self.auv.bodyPoints, referenceTranslation)
        self.bodyPoints.set_data_3d(*zip(*bodyPoints))
        self.trail.set_data_3d(*zip(*[np.array(t) - np.array(referenceTranslation) for t in self.auv.trail[0::20]]))

        return self.artists()

if __name__ == "__main__":
    pass
    


