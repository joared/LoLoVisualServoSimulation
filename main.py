
if __name__ == "__main__":
    import numpy as np
    from camera import Camera, FeatureSet
    from animation import CameraAnimator
    # instable for LeStar
    #camera = Camera(translation=(-5, -1, -2), euler=(-0.3, -0.2, 0.1))

    camera = Camera(translation=(-5, -0.9, -1.5), euler=(-.5, -0.1, 0.1))
    # feature positions
    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), euler=(0, -np.pi/2, 0))
    # desired position of feature in image plane (y [left], z [up])
    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]

    #cameraPlotter = CameraPLotter(camera, features, targets)
    cameraAnimator = CameraAnimator(camera, featureSet.transformedFeatures(), targets, None)
    cameraAnimator.animate()
    #cameraAnimator.anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    cameraAnimator.show()