import numpy as np
from camera import Camera, FeatureSet
from animation import CameraAnimator

def triangleTest(**cameraArgs):
    camera = Camera(translation=(-5, -0.9, -1.5), 
                    euler=(-.5, -0.1, 0.1), 
                    **cameraArgs)

    featureSet = FeatureSet([[-1, 0, 0], [0, 0, 1], [1, 0, 0]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/2, 0))
    targets = [[0.3, 0], [0, 0.3], [-0.3, 0]]

    return camera, featureSet, targets

def stressTest(**cameraArgs):
    camera = Camera(translation=(-6, -1, -0.4), 
                    euler=(-.55, 0.2, 0.2), 
    #camera = Camera(translation=(-5, -1.2, -1.5),
    #                euler=(-.5, -0.1, 0.1), 
                    **cameraArgs)

    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/2, 0))

    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]

    return camera, featureSet, targets

def rollTest(**cameraArgs):
    camera = Camera(translation=(0, -1/0.3, 0), 
                    euler=(0, 0, np.pi/2),
                    **cameraArgs)

    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/4, 0))
    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]

    return camera, featureSet, targets

def yawTest(**cameraArgs):
    camera = Camera(translation=(0, -1/0.3, 0), 
                    euler=(0, 0, np.pi/2.7),
                    **cameraArgs)

    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), 
                            euler=(0, 0, 0))
    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]

    return camera, featureSet, targets

def translationTest(*cameraArgs):
    camera = Camera(translation=(-8, -2, 0), 
                    euler=(0, 0, np.pi/4),
                    **cameraArgs)

    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), 
                            euler=(0, 0, -np.pi/4))

    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]
    return camera, featureSet, targets

def normalTest(**cameraArgs):
    camera = Camera(translation=(3.5, -3/0.3, 0), 
                    euler=(0, 0, np.pi/2*0.9),
                    **cameraArgs)

    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/4, 0))
    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]

    return camera, featureSet, targets

if __name__ == "__main__":
    # features: feature positions
    # targets: desired position of feature in image plane (y [left], z [up])
    # instable for LeStar
    #camera = Camera(translation=(-5, -1, -2), euler=(-0.3, -0.2, 0.1))
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--controller', '-c', type=str, default="IBVS1",
                        help='controller: IBVS1, IBVS2, IBVS3, PBVS1, PBVS2')
    #parser.add_argument('--rule', '-r', type=str,
    #                    help='control rule for camera motion (Le, LeStar or LeLeStar)')
    parser.add_argument('--scenario', '-s', default="normal",
                        help='choose camera and feature scenario')
    parser.add_argument('--lamb', '-l', type=float, default=0.01,
                        help='exponential error decay')
    parser.add_argument('--noiseStd', '-n', type=float, default=0,
                        help='noise standard deviation of projected features')

    args = parser.parse_args()
    print(args)

    cameraArgs = {"controller": args.controller, 
                  "lamb": args.lamb, 
                  "noiseStd": args.noiseStd}

    tests = (normalTest,
             translationTest,
             rollTest,
             yawTest,
             triangleTest,
             stressTest)

    dictTests = {t.__name__: t for t in tests}
    camera, featureSet, targets = dictTests[args.scenario + "Test"](**cameraArgs)
    """
    if args.scenario == "normal":
        camera, featureSet, targets = normalTest(**cameraArgs)
    elif args.scenario == "translation":
        camera, featureSet, targets = translationTest(**cameraArgs)
    elif args.scenario == "roll":
        camera, featureSet, targets = rollTest(**cameraArgs)
    elif args.scenario == "yaw":
        camera, featureSet, targets = yawTest(**cameraArgs)
    elif args.scenario == "triangle":
        camera, featureSet, targets = triangleTest(**cameraArgs)
    elif args.scenario == "stress":
        camera, featureSet, targets = stressTest(**cameraArgs)
    else:
        raise Exception("Invalid scenario '{}'".format(args.scenario))
    """
    cameraAnimator = CameraAnimator(camera, featureSet, targets, None)
    cameraAnimator.animate()
    #cameraAnimator.anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    cameraAnimator.show()