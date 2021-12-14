import torch
import numpy as np
from pytorch3d import transforms as p3d_transforms
from scipy.spatial.transform import Rotation


def np_dcm2euler(mats: np.ndarray, seq: str = "zyx", degrees: bool = True):
    """Converts rotation matrix to euler angles

    Args:
        mats: (B, 3, 3) containing the B rotation matricecs
        seq: Sequence of euler rotations (default: "zyx")
        degrees (bool): If true (default), will return in degrees instead of radians

    Returns:

    """

    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=degrees))
    return np.stack(eulers)


def np_transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SO3 transform

    Args:
        g: SO3 transformation matrix of size (B, 3, 3)
        pts: Points to be transformed (B, N, 3)

    Returns:
        transformed points of size (B, N, 3)

    """
    rot = g[..., :3, :3]  # (3, 3)
    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2)
    return transformed


def np_inverse(g: np.ndarray):
    """Returns the inverse of the SE3 transform

    Args:
        g: ([B,] 3/4, 4) transform

    Returns:
        ([B,] 3/4, 4) matrix containing the inverse

    """
    rot = g[..., :3, :3]  # (3, 3)

    inv_rot = np.swapaxes(rot, -1, -2)

    return inv_rot


def torch_dcm2euler(mats, seq, degrees=True):
    if seq == "xyz":
        eulers = p3d_transforms.matrix_to_euler_angles(mats, "ZYX")
    elif seq == "zyx":
        eulers = p3d_transforms.matrix_to_euler_angles(mats, "XYZ")
    eulers = eulers[:, [2, 1, 0]]
    if degrees:
        eulers = eulers / np.pi * 180
    return eulers


def torch_quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy,
        2 * wx + 2 * yz, w2 - x2 - y2 + z2
    ],
                         dim=1).reshape(B, 3, 3)
    return rotMat


if __name__ == "__main__":
    anglex = np.pi / 2
    angley = np.pi / 2
    anglez = 0
    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

    np_mat = (Rx @ Ry @ Rz)[None, ...]
    torch_mat = torch.from_numpy(np_mat)

    np_inv_mat = np_inverse(np_mat)

    np_euler = np_dcm2euler(np_inv_mat, "xyz")
    torch_euler = torch_dcm2euler(torch_mat, "xyz")
    print("=" * 50)
    print(np_euler)
    print(torch_euler)

    # src = np.array([[[1, 0, 0]]])
    # ref_forward = np_transform(np_mat, src)
    # print(ref_forward)
    # ref_backword = np_transform(np_inv_mat, ref_forward)
    # print(ref_backword)
