from scipy.spatial.transform import Rotation as R
import numpy as np

from coordinate_system import CoordinateSystem, CoordinateSystemArtist

class Perception:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, camera, featureModel):
        self.camera = camera
        self.camera.
        self.featureModel = featureModel

    def processImage(self, img):
        pass

    def associate(self, featurePoints):
        pass

    def estimatePose(self, imagePoints):
        pass

    def routine(self, img):
        pass

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
        self.trail = ax.plot3D([], [], [], color="goldenrod", marker="^", linewidth=1)[0]

        return self.artists()

    def update(self, showAxis=True, referenceTranslation=(0,0,0)):
        CoordinateSystemArtist.update(self, showAxis, referenceTranslation)

        bodyPoints = self.auv.transformedPoints(self.auv.bodyPoints, referenceTranslation)
        self.bodyPoints.set_data_3d(*zip(*bodyPoints))
        self.trail.set_data_3d(*zip(*[np.array(t) - np.array(referenceTranslation) for t in self.auv.trail[0::20]]))

        return self.artists()

if __name__ == "__main__":
    pass
    


