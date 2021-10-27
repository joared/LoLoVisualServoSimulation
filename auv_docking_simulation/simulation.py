import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

from camera import Camera, CameraArtist
from auv import AUV, AUVArtist
from feature import FeatureModel, FeatureModelArtist3D

from scipy.spatial.transform import Rotation as R

class Simulator:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.camera = Camera(np.eye(3))
        self.auv = AUV(self.camera, 
                       "back",
                       size=(3, 10),
                       euler=(0, 0, 0))

        self.feature = FeatureModel(1, 8, shift=False, zShift=0)
        self.dockingStation = AUV(self.feature,
                                  "front",
                                  size=(10, 10),
                                  translation=[-15, 15, 4],
                                  euler=(0, 0, 0))

        self.poseEstimator = None

        self.cameraArtist = CameraArtist(self.camera, len(self.feature.points))
        self.auvArtist = AUVArtist(self.auv)
        self.featureArtist = FeatureModelArtist3D(self.feature)
        self.dockingStationArtist = AUVArtist(self.dockingStation, color="navy")

        # Maybe this should only be one artist??
        #self.poseEstimationArtist = PoseEstimationArtist(self.camera, self.feature)

        self.focusCoord = self.auv

        self.callback = None
        self.blit = False
        self.showAxis = True
        self.updateCenterNr = 200

    def on_press(self, event):
        print('press', event.key)
        if event.key == 'x':
            self.showAxis = not self.showAxis
        elif event.key == "c":
            if self.focusCoord == self.auv:
                self.focusCoord = self.camera
            elif self.focusCoord == self.camera:
                self.focusCoord = self.dockingStation
            elif self.focusCoord == self.dockingStation:
                self.focusCoord = self.feature
            else:
                self.focusCoord = self.auv
            self.setAxisLims()
            #self.fig.canvas.draw()
        elif event.key == "up":
            self.size = max(self.size - 1, 1)
            self.setAxisLims()
            self.fig.canvas.draw()
        elif event.key == "down":
            self.size = self.size + 1
            self.setAxisLims()
            self.fig.canvas.draw()

    def setAxisLims(self):
        center = self.focusCoord.translation
        if self.centerAxis:
            center = (0, 0, 0)
        
        self.ax.set_xlim3d(-self.size + center[0], self.size + center[0])
        self.ax.set_ylim3d(-self.size + center[1], self.size + center[1])
        self.ax.set_zlim3d(-self.size + center[2], self.size + center[2])            
        
        center = (0, 0, 0)
        if self.centerAxis:
            center = self.focusCoord.translation
        
        if self.blit is True and not self.centerAxis: # is this correct?
        #if self.blit is True:
            self.fig.canvas.draw()

        return center

    def _setAxisLims(self):
        center = self.focusCoord.translation
        
        if self.blit is True:
            if self.centerAxis:
                center = (0, 0, 0)
            self.ax.set_xlim3d(-self.size + center[0], self.size + center[0])
            self.ax.set_ylim3d(-self.size + center[1], self.size + center[1])
            self.ax.set_zlim3d(-self.size + center[2], self.size + center[2])            
            self.fig.canvas.draw()
        else:
            self.ax.set_xlim3d(-self.size + center[0], self.size + center[0])
            self.ax.set_ylim3d(-self.size + center[1], self.size + center[1])
            self.ax.set_zlim3d(-self.size + center[2], self.size + center[2])
        
            center = (0, 0, 0)
                
        if not self.centerAxis:
            center = (0, 0, 0)

        return center

    def animate(self, callback=None, anim=True, blit=False, centerAxis=False):
        self.callback = callback
        self.blit = blit
        self.centerAxis = centerAxis

        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

        gs = self.fig.add_gridspec(2, 2)

        self.ax = self.fig.add_subplot(gs[:, :], projection="3d")
        
        self.ax.set_title("Docking simulator")
        self.size = 15
        # set_aspect("equal") doesn't exist for 3D axes yet
        self.ax.set_xlim3d(-self.size, self.size)
        self.ax.set_ylim3d(-self.size, self.size)
        self.ax.set_zlim3d(-self.size, self.size)

        self.startTime = 0
        self.elaspsed = 0

        # need to store FuncAnimation otherwise it doesn't work
        if anim:
            self.anim = animation.FuncAnimation(self.fig, self.update, frames=self.timeGen, init_func=self.init,
                                                interval=self.dt*1000, blit=blit)
        
        self.fig.canvas.draw() # bug fix

    def show(self):
        plt.show()

    def timeGen(self):
        #for i in range(1500):
        i = 0
        while True:
            yield i
            i += 1

    def init(self):
        return self.cameraArtist.init(self.ax) + self.auvArtist.init(self.ax) + self.featureArtist.init(self.ax) + self.dockingStationArtist.init(self.ax) #+ [self.ax.spines["left"], self.ax.spines["right"], self.ax.spines["top"], self.ax.spines["bottom"]] + [self.xAxis, self.yAxis, self.zAxis]

    def update(self, i):
        if i == 0:
            self.startTime = time.time()
        self.elapsed = time.time() - self.startTime

        ###############################
        if self.callback:
            self.callback(i, self.dt)
        """
        sim = self
        if i % 100 == 0:
            if sim.focusCoord == sim.auv:
                sim.focusCoord = sim.camera
            else:
                sim.focusCoord = sim.auv

        vel = [0.1, 0, 0, 0, 0, 0.03]

        sim.auv.move(vel)
        """
        ###############################
        center = (0, 0, 0)
        if self.centerAxis:
            center = self.focusCoord.translation
        if i % self.updateCenterNr == 0:
            refPoint = np.array(self.feature.transformedPoints([[0, 0, 3]])[0])
            #plt.gca().scatter(*refPoint)
            self.setAxisLims()
        
        
        #self.xAxis.set_ticks(list(range(int(center[0]-3), int(center[0]+3))))
        #self.yAxis.set_ticks(list(range(int(center[1]-3), int(center[1]+3))))

        # bug fix, title/info text not updating otherwise
        # however, really slows down simulation (doesn't work with blitting)
        # self.fig.canvas.draw()
        showAxis = self.showAxis
        return self.cameraArtist.update(showAxis, center) + \
               self.auvArtist.update(showAxis, center) + \
               self.featureArtist.update(showAxis, center) + \
               self.dockingStationArtist.update(showAxis, center) #+ [self.ax.spines["left"], self.ax.spines["right"], self.ax.spines["top"], self.ax.spines["bottom"]] + [self.xAxis, self.yAxis, self.zAxis]

def control(i, dt):
    #print("Time:", i*dt)

    if i% 5 == 0:
        featurePoints = np.array(sim.feature.transformedPoints(sim.feature.points))
        noise = np.random.normal(0, 2*0.0000028, featurePoints.shape)
        sim.camera.detectFeatures(list(featurePoints + noise))
    else:
        sim.camera.detectFeatures([]) # removes detection reference in plot

    vel = [1.5, 0, 0, 0, 0, 0.006]
    sim.dockingStation.move(vel, dt)

    global velAUV, errPrev, angleErrPrev

    refPoint = np.array(sim.feature.transformedPoints([[0, 0, 3]])[0])
    trans = refPoint - np.array(sim.auv.translation)

    err = np.dot(np.array(sim.auv.rotation)[:, 0], trans)
    #print("Error:", err)
    accelAUV = 1*err
    #accelAUV = min(accelAUV, 0.1)
    #accelAUV = 0.005*l
    
    accelAUV = accelAUV + 40*(err-errPrev)
    accelAUVMax = 0.1
    accelAUV = min(accelAUV, accelAUVMax)
    accelAUV = max(accelAUV, -accelAUVMax)
    #print(accelAUV)
    errPrev = err
    vMax = 100#2
    vx = min(velAUV[0] + accelAUV, vMax)
    vMin = 0.5
    vMax = 3
    vx = max(vx, vMin)
    vx = min(vx, vMax)
    velAUV[0] = vx

    l = np.linalg.norm(trans)
    rotRefLength = 3
    featToAUV = np.array(sim.auv.translation).transpose()-np.array(sim.feature.translation).transpose()
    projAUVOnFeatureLine = np.dot(featToAUV.transpose(), np.array(sim.feature.rotation)[:, 2])
    xDir = (projAUVOnFeatureLine+rotRefLength)*np.array(sim.feature.rotation)[:, 2] + np.array(sim.feature.translation).transpose() - np.array(sim.auv.translation).transpose()
    xDir = xDir/np.linalg.norm(xDir)
    xDir = xDir.transpose()
    zDir = np.array([0, 0, 1])
    yDir = np.cross(zDir, xDir)
    yDir /= np.linalg.norm(yDir)
    zDir = np.cross(xDir, yDir)
    Rp = np.column_stack( (xDir.transpose(), yDir.transpose(), zDir.transpose()) )
    Rc = np.array(sim.auv.rotation)
    w = R.from_matrix(np.matmul(np.linalg.inv(Rc), Rp)).as_rotvec()
    Rw = R.from_rotvec(w).as_matrix()
    Rerrprev = R.from_rotvec(angleErrPrev).as_matrix()

    wdt = R.from_matrix(np.matmul(np.linalg.inv(Rerrprev), Rw)).as_rotvec() # think this is the one, but both acts similar
    #wdt = R.from_matrix(np.matmul(Rw, np.linalg.inv(Rerrprev))).as_rotvec()

    angleErrPrev = w
    angMax = 0.2
    angVel = w*0.2 + wdt*20
    if np.linalg.norm(angVel) > angMax:
        angVel = angVel/np.linalg.norm(angVel)*angMax
    velAUV[3:] = list(angVel)

    sim.auv.move(velAUV, dt)
    

if __name__ == "__main__":
    dt = 0.02 # changing dt changes the behaviiour of the camera detection frequency (needs to be fixed)
    sim = Simulator(dt)
    errPrev = 0
    angleErrPrev = [0, 0, 0]
    velAUV = [1.5, 0, 0, 0, 0, 0]
    sim.animate(control, anim=True, blit=True, centerAxis=True)
    sim.show()
    exit()

    sim.animate(None, anim=False, blit=True, centerAxis=False)
    sim.init()
    
    for i in range(1000):
        
        ##################################
        control(i, dt)

        ###################################
        sim.update(i)
        plt.pause(dt)

    #sim.animate()
    sim.show()
