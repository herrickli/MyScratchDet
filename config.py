# config.py
import os.path


class config:
    # gets home dir cross platform
    HOME = os.path.expanduser("~")

    VOC_CLASSES = ('__BACKGROUND__',  # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    NUM_CLASSES = 21

    INPUT_IMAGE_SIZE = 512

    BATCH_SIZE = 2

    # for making bounding boxes pretty
    COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
              (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
    MEANS = (104, 117, 123)

    # SSD CONFIG
    NUM_DEFAULT_BOX = [4, 6, 6, 6, 4, 4]

    FEATURE_MAP_STRIDE = [8, 16, 32, 64, 85, 128]

    ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    OUT_CHANNELS = [1024, 2048, 512, 256, 256, 256]

    OUT_FEATURE_MAP_SIZE = [64, 32, 16, 8, 6, 4]

    BASE_SIZE_RATIO_RANGE = [0.1, 0.9]

    PRIOR_BOX_MIN_SIZE = [25.6, 51.2, 153.6, 256, 358.4, 460.8]

    PRIOR_BOX_MAX_SIZE = [51.2, 153.6, 256, 358.4, 460.8, 563.2]

    PRIOR_BOX_CLIP = True

    # SSD300 CONFIGS
    voc = {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC',
    }

    coco = {
        'num_classes': 201,
        'lr_steps': (280000, 360000, 400000),
        'max_iter': 400000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [21, 45, 99, 153, 207, 261],
        'max_sizes': [45, 99, 153, 207, 261, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'COCO',
    }


cfg = config
