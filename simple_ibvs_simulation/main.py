import numpy as np
from camera import Camera, FeatureSet
from animation import CameraAnimator

def triangle(controller, controlRule):
    camera = Camera(translation=(-5, -0.9, -1.5), 
                    euler=(-.5, -0.1, 0.1), 
                    controller=controller,
                    controlRule=controlRule)
    featureSet = FeatureSet([[-1, 0, 0], [0, 0, 1], [1, 0, 0]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/2, 0))
    targets = [[0.3, 0], [0, 0.3], [-0.3, 0]]
    return camera, featureSet, targets

def stressTest(controller, controlRule):
    camera = Camera(translation=(-6, -1, -0.4), 
                    euler=(-.55, 0.2, 0.2), 
    #camera = Camera(translation=(-5, -1.2, -1.5),
    #                euler=(-.5, -0.1, 0.1), 
                    controller=controller,
                    controlRule=controlRule)
    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/2, 0))
    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]

    return camera, featureSet, targets

def rotationTest(controller, controlRule):
    camera = Camera(translation=(0, -1/0.3, 0), 
                    euler=(0, 0, np.pi/2),
                    controller=controller,
                    controlRule=controlRule)
    featureSet = FeatureSet([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]], 
                            translation=(0, 0, 0), 
                            euler=(0, -np.pi/4, 0))
    targets = [[-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]]
    return camera, featureSet, targets

def normalTest(controller, controlRule):
    camera = Camera(translation=(3.5, -3/0.3, 0), 
                    euler=(0, 0, np.pi/2*0.9),
                    controller=controller,
                    controlRule=controlRule)
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
    parser.add_argument('--controller', '-c', type=str, default="IBVS",
                        help='controller: IBVS or PBVS')
    parser.add_argument('--rule', '-r', type=str, default="Le",
                        help='control rule for camera motion (Le, LeStar or LeLeStar)')
    parser.add_argument('--scenario', '-s', default="normal",
                        help='choose camera and feature scenario')

    args = parser.parse_args()
    print(args)

    controller = args.controller
    controlRule = args.rule
    if args.scenario == "normal":
        camera, featureSet, targets = normalTest(controller, controlRule)
    elif args.scenario == "rot":
        camera, featureSet, targets = rotationTest(controller, controlRule)
    elif args.scenario == "triangle":
        camera, featureSet, targets = triangle(controller, controlRule)
    elif args.scenario == "stress":
        camera, featureSet, targets = stressTest(controller, controlRule)
    else:
        raise Exception("Invalid scenario '{}'".format(args.scenario))
    
    cameraAnimator = CameraAnimator(camera, featureSet.transformedFeatures(), targets, None)
    cameraAnimator.animate()
    #cameraAnimator.anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    cameraAnimator.show()