import mujoco
import numpy as np


def xmat_to_quat_xyzw(xmat):
    """
    Convert a 3x3 rotation matrix (flattened, as in MuJoCo xmat) to a quaternion [x, y, z, w].
    MuJoCo returns [w, x, y, z], so we reorder to [x, y, z, w].
    """
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.array(xmat).reshape(9))
    # MuJoCo: [w, x, y, z] -> [x, y, z, w]
    quat = np.roll(quat, shift=-1, axis=-1)
    return quat
