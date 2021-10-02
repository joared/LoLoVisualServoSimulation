import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

from camera import Camera, CameraArtist
from auv import AUV, AUVArtist

from scipy.spatial.transform import Rotation as R

class Simulator:
    def __init__(self):
        self.camera = Camera(np.eye(3))
        self.auv = AUV(self.camera)
        #self.dockingStation = None

        self.cameraArtist = CameraArtist(self.camera)
        self.auvArtist = AUVArtist(self.auv)
        #self.dockingStationArtist = DockingStationArtist(self.dockingStation)

        self.focusCoord = self.auv

    def animate(self, anim=True, blit=False):
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(2, 2)

        self.ax = self.fig.add_subplot(gs[:, :], projection="3d")
        
        self.ax.set_title("Docking simulator")
        self.size = 5
        # set_aspect("equal") doesn't exist for 3D axes yet
        self.ax.set_xlim3d(-self.size, self.size)
        self.ax.set_ylim3d(-self.size, self.size)
        self.ax.set_zlim3d(-self.size, self.size)

        self.startTime = 0
        self.elaspsed = 0

        # need to store FuncAnimation otherwise it doesn't work
        if anim:
            self.anim = animation.FuncAnimation(self.fig, self.update, frames=self.timeGen, init_func=self.init,
                                                interval=0, blit=blit)
        
        self.fig.canvas.draw() # bug fix

    def show(self):
        plt.show()

    def timeGen(self):
        for i in range(1000):
            yield i

    def init(self):
        return self.cameraArtist.init(self.ax) + self.auvArtist.init(self.ax)

    def update(self, i):
        if i == 0:
            self.startTime = time.time()
        self.elapsed = time.time() - self.startTime

        ###############################
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

        center = self.focusCoord.translation
        # set_aspect("equal") doesn't exist for 3D axes yet
        self.ax.set_xlim3d(-self.size + center[0], self.size + center[0])
        self.ax.set_ylim3d(-self.size + center[1], self.size + center[1])
        self.ax.set_zlim3d(-self.size + center[2], self.size + center[2])

        # bug fix, title/info text not updating otherwise
        # however, really slows down simulation (doesn't work with blitting)
        #self.fig.canvas.draw()
        return self.cameraArtist.update() + self.auvArtist.update()

if __name__ == "__main__":
    sim = Simulator()

    sim.animate(anim=False, blit=True)
    sim.init()
    for i in range(1000):
        ##################################
        if i % 100 == 0:
            if sim.focusCoord == sim.auv:
                sim.focusCoord = sim.camera
            else:
                sim.focusCoord = sim.auv

        vel = [0.1, 0, 0, 0, 0, 0.03]

        sim.auv.move(vel)

        ###################################
        sim.update(i)
        plt.pause(0.001)

    #sim.animate()
    sim.show()