import numpy as np

from coordinate_system import CoordinateSystem, CoordinateSystemArtist

def polygon(rad, n, shift=False, zShift=0):
    theta = 2*np.pi/n
    if shift is True:
        #points = np.array([[0, 0, 0, 1]] + [ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), 0, 1] for i in range(n)], dtype=np.float32)
        points = np.array([ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), zShift, 1] for i in range(n)] , dtype=np.float32)
    else:
        points = np.array([ [rad*np.sin(theta*i), rad*np.cos(theta*i), zShift, 1] for i in range(n)], dtype=np.float32)

    return points

class FeatureModel(CoordinateSystem):
    def __init__(self, rad, n, shift=False, zShift=0, *args, **kwargs):
        CoordinateSystem.__init__(self, *args, **kwargs)
        self.points = list(polygon(rad, n, shift, zShift)[:, :3]) # homogenious not needed

class FeatureModelArtist3D(CoordinateSystemArtist):
    def __init__(self, feature, *args, **kwargs):
        CoordinateSystemArtist.__init__(self, feature, *args, **kwargs)
        self.feature = feature
        
        self.features3D = None
        #self.target3D = None    
        #self.featuresProj3D = None
        #self.referenceLines = []

    def artists(self):
        return [self.features3D] + CoordinateSystemArtist.artists(self)

    def init(self, ax):
        CoordinateSystemArtist.init(self, ax)
        self.features3D = ax.plot3D([], [], [], marker="o", color="m")[0]
        #self.features3D = ax.scatter([], [], [], color="navy")

        return self.artists()

    def update(self, showAxis=True, referenceTranslation=(0,0,0)):
        CoordinateSystemArtist.update(self, showAxis, referenceTranslation)
        points = self.feature.transformedPoints(self.feature.points, referenceTranslation)
        self.features3D.set_data_3d(*zip(*points + [points[0]])) # line
        #self.features3D._offsets3d = [*zip(*points + [points[0]])] # scatter

        return self.artists()

class CameraAnimator:
    def __init__(self, camera, features, targets, controlCallback):
        self.camera = camera
        self.features = features
        self.targets = targets
        self.controlCallback = controlCallback
        # camera trajectory
        self.cameraPositions = []
        # feature trajectories
        self.featurePositions = [[] for _ in features]
        # errors
        self.velocities = []

        self.ax = None
        self.ax2 = None
        self.timeStep = -1

        self.focalLine = None
        self.topLine = None # Reference for camera orientation
        self.cameraLines = None
        self.imagePlane = None
        self.target3D = None
        self.target2D = None
        self.features3D = None
        self.cameraTrajectory = None
        self.featuresProj3D = None
        self.referenceLines = []
        self.featureTrajectories = []
        self.features2D = None
        self.velocityLines = []

        self.play = False

    def artists(self):
        artists = [self.focalLine, 
                   self.topLine, 
                   self.cameraLines, 
                   self.imagePlane,
                   self.target3D,
                   self.target2D,
                   self.features3D, 
                   self.cameraTrajectory,
                   self.featuresProj3D,
                   *self.referenceLines,
                   *self.featureTrajectories,
                   self.features2D,
                   *self.velocityLines]

        return [a for a in artists if a is not None]

    def on_clicked(self, event):
        self.play = not self.play

    def animate(self):

        #self.fig = plt.figure(constrained_layout=True)
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(2, 2)

        #self.ax = plt.subplot(1, 2, 1, projection="3d")
        self.ax = self.fig.add_subplot(gs[1, :], projection="3d")
        size = 5
        # set_aspect("equal") doesn't exist for 3D axes yet
        self.ax.set_title("Camera motion")
        self.ax.set_xlim3d(-size, size)
        self.ax.set_ylim3d(-size, size)
        self.ax.set_zlim3d(-size, size)

        #self.info_text = self.ax.text(0, 0, 0, 'time info', horizontalalignment="left",
        #                              verticalalignment="bottom", transform=self.ax.transAxes)
        #self.ax2 = plt.subplot(1, 2, 2)
        self.ax2 = self.fig.add_subplot(gs[0, 0])
        self.ax2.clear()
        self.ax2.set_title("Image plane")
        self.ax2.set_aspect("equal")
        size = self.camera.imSize/2*1.1 # a little bigger
        self.ax2.set_xlim(-size, size)
        self.ax2.set_ylim(-size, size)

        self.ax3 = self.fig.add_subplot(gs[0, 1])
        self.ax3.set_title('Velocity (vx, vy, vz, wx, wy, wz)')
        self.ax3.set_xlim(0, 1000)
        self.ax3.set_ylim(-self.camera.imSize/2, self.camera.imSize/2)
        self.playButton = Button(self.ax3, "Press to play/pause")
        self.playButton.on_clicked(self.on_clicked)

        self.startTime = 0
        self.elaspsed = 0

        # need to store FuncAnimation otherwise it doesn't work
        self.anim = animation.FuncAnimation(self.fig, self.update, frames=self.timeGen, init_func=self.init,
                                            interval=0, blit=True)
        
        self.fig.canvas.draw() # bug fix

    def show(self):
        plt.show()

    def timeGen(self):
        err = [np.inf]
        threshold = 0.001
        i = 0
        while not all([e < threshold for e in err]):
            yield i
            v, err = self.camera.control(self.targets, self.features, lamb=0.01)
            i += 1
        if self.play == False:
            self.cameraPositions = []
            self.featurePositions = [[] for _ in self.features]
            self.velocities = []
            self.camera.reset()

    def init(self):
        # init is called twice, need to reset lists
        self.referenceLines = []
        self.featureTrajectories = []

        # plot camera
        points = self.camera.transformedPoints()
        line = self.camera.transformedFocalLine()

        self.focalLine = self.ax.plot3D([], [], [], color="black")[0]
        self.topLine = self.ax.plot3D([], [], [], color="black")[0]
        self.cameraLines = self.ax.plot3D([], [], [], color="r")[0]

        # plot image plane (static)
        #self.imagePlane = self.ax2.plot([1, 1, -1, -1, 1], [1, -1, -1, 1, 1], "r")[0]
        self.imagePlane = self.ax2.plot(*zip(*[(p[1], p[2]) for p in self.camera.points + [self.camera.points[0]]]), "r")[0]

        # plot targets
        self.target3D = self.ax.plot3D([], [], [], color="grey", marker=".")[0]
        # 2D static
        self.target2D = self.ax2.plot(*[*zip(*self.targets + [self.targets[0]])], color="grey", marker="o")[0]
        
        # plot features
        self.features3D = self.ax.plot3D([], [], [], marker="o", color="navy")[0]

        # plot camera trajectory
        self.cameraTrajectory = self.ax.plot3D([], [], [], color="g")[0]
                
        projPoints = [self.camera.project(feature) for feature in self.features]
        # TODO: check if projPopint is False
        self.featuresProj3D = self.ax.plot3D([], [], [], color="cornflowerblue", marker=".")[0]
        for pp, f in zip(projPoints, self.features):
            self.referenceLines.append(self.ax.plot3D([], [], [], color="grey")[0])

        # plot feature trajectory
        colors = ["lightcoral", "cyan", "navy", "black", "magenta", "yellow", "brown", "lightblue"] # hardcoded, might be more than 4 features!!!
        for feature, color in zip(self.features, colors):
            #self.featureTrajectories.append(self.ax2.plot([], [], color="lightcoral", linestyle="dashed")[0])
            self.featureTrajectories.append(self.ax2.plot([], [], color=color, linestyle="dashed")[0])

        # plot 2D projected features in image plane
        self.features2D = self.ax2.plot([], [], color="navy", marker="o")[0]

        # plot velocities
        self.velocityLines = [self.ax3.plot([], [], linestyle="dashed", color=colors[i])[0] for i in range(6)]
        self.ax3.legend(self.velocityLines, ["vx", "vy", "vz", "wx", "wy", "wz"])
 
        return self.artists()

    def update(self, i):
        if i == 0:
            self.startTime = time.time()
        self.elapsed = time.time() - self.startTime

        if self.play:
            #self.controlCallback()
            v, err = self.camera.control(self.targets, self.features, lamb=0.01)    
            threshold = 0.001
            if not all([e < threshold for e in err]):
                if v is not False:
                    self.camera.move(v)
                    self.velocities.append(v)
                    self.cameraPositions.append(np.array(self.camera.translation))
                    # Add feature trajectory/positions
                    for feature, featPos in zip(self.features, self.featurePositions):
                        projPoint = self.camera.globalToImage(feature)
                        featPos.append((-projPoint[0], projPoint[1]))

        # plot camera
        points = self.camera.transformedPoints()
        line = self.camera.transformedFocalLine()
        self.focalLine.set_data_3d(*zip(*line))
        self.topLine.set_data_3d(*zip(*self.camera.transformedTopLine()))
        self.cameraLines.set_data_3d(*zip(*points + [points[0]]))

        # plot image plane
        #self.imagePlane.set_data([1, 1, -1, -1, 1], [1, -1, -1, 1, 1]) # static

        # plot targets
        points = [self.camera.imageToGlobal(target) for target in self.targets]
        self.target3D.set_data_3d(tuple([*zip(*points + [points[0]])]))
        #self.target2D.set_offsets(np.array([*zip(*targets)]).transpose()) # static

        # plot features
        self.features3D.set_data_3d(*zip(*self.features + [self.features[0]]))

        # plot camera trajectory
        if self.cameraPositions:
            self.cameraTrajectory.set_data_3d(*zip(*self.cameraPositions))

        # plot 3D projected features in image plane 
        projPoints = [self.camera.project(feature) for feature in self.features]
        self.featuresProj3D.set_data_3d(*zip(*projPoints + [projPoints[0]]))

        # plot reference lines
        for l, pp, f in zip(self.referenceLines, projPoints, self.features):
            l.set_data_3d(*zip(pp, f))

        # plot feature trajectory
        if self.featurePositions[0]:
            for featTraj, featPos in zip(self.featureTrajectories, self.featurePositions):
                featTraj.set_data(*zip(*featPos))

        # plot 2D projected features in image plane
        imPoints = [self.camera.globalToImage(feature) for feature in self.features]
        imPoints = [(-point[0], point[1]) for point in imPoints]
        self.features2D.set_data(*zip(*imPoints + [imPoints[0]]))
        
        # plot velocities
        if self.velocities:
            for i, velLine in enumerate(self.velocityLines):
                velLine.set_data(*zip(*[(t, vel[i]) for t, vel in enumerate(self.velocities)]))
            self.ax3.set_xlim(0, len(self.velocities))
            m = max(abs(np.ndarray.flatten(np.array(self.velocities))))
            self.ax3.set_ylim(-m*1.1, m*1.1)

        # bug fix, title/info text not updating otherwise
        # however, really slows down simulation (doesn't work with blitting)
        #self.fig.canvas.draw()
        return self.artists()

        