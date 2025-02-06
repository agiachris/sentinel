import numpy as np


class hardware:
    ARM_MOUNTING_HEIGHT = 0.335  # 0.335 for mobile base, 0.02 for table top
    ARM_REACH_APPROX = 1.5
    BASE_SAFE_MARGIN = 0.15
    BASE_VEL_THRESH = 0.05
    MIN_DIST_TO_FLOOR = 0.005
    MIN_EE_DIST = 0.20
    EE_XY_LOW = np.array([0.45, -0.15])
    EE_XY_HIGH = np.array([0.75, 0.15])
    BASE_TGT_LOWS = np.array([-1.5, -1.2])
    BASE_TGT_HIGHS = np.array([1.7, 1.2])

    BASE_DIFF_THRESH = np.array([0.015, 0.015, 0.02])
    BASE_MOVE_TIMEOUT = 10.0
    BASE_STEPS = 200


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
