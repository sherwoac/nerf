import numpy as np
import math
import mathutils


def rotate_about_X(T, elevation_deg):
    elevation = elevation_deg * np.pi / 180.
    Rx = np.array([
        [1., 0, 0, 0],
        [0, np.cos(elevation), -np.sin(elevation), 0],
        [0, np.sin(elevation), np.cos(elevation), 0],
        [0, 0, 0, 1.],
    ], dtype=np.float32)  # rotation by elevation

    return np.dot(Rx, T)


def rotate_about_Z(T, azimuth_deg):
    azimuth = azimuth_deg * np.pi / 180.
    Rz = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0, 0],
        [np.sin(azimuth), np.cos(azimuth), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])  # rotation by azimuth

    return np.dot(Rz, T)


def translate_X(T, translation_m):
    T_copy = np.copy(T)
    T_copy[:, 3] += np.array([translation_m, 0., 0., 1.])
    return T_copy


def translate_Y(T, translation_m):
    T_copy = np.copy(T)
    T_copy[:, 3] += np.array([0., translation_m, 0., 1.])
    return T_copy


def translate_Z(T, translation_m):
    T_copy = np.copy(T)
    T_copy[:, 3] += np.array([0., 0., translation_m, 1.])
    return T_copy


def get_transformation_matrix(azimuth, elevation, distance):
    if distance == 0:
        return np.empty((4, 4))

    # camera center
    C = np.zeros((3, 1))
    C[0] = distance * np.cos(elevation) * np.sin(azimuth)
    C[1] = -distance * np.cos(elevation) * np.cos(azimuth)
    C[2] = distance * np.sin(elevation)

    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = azimuth_convention(azimuth)
    elevation = elevation_convention(elevation)

    # rotation matrix
    Rz = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1],
    ])  # rotation by azimuth
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(elevation), -np.sin(elevation)],
        [0, np.sin(elevation), np.cos(elevation)],
    ])  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    t= np.dot(-R_rot, C)
    return R_t_to_R(R_rot, t)
    # R = np.hstack((R_rot, np.dot(-R_rot, C)))
    # R = np.vstack((R, [0, 0, 0, 1]))
    # return R

def R_t_to_R(R, t):
    R = np.hstack((R, t))
    return np.vstack((R, [0, 0, 0, 1]))


def azimuth_convention(azimuth):
    return -azimuth


def elevation_convention(elevation):
    return elevation - 0.5 * np.pi


def azimuth_unconvention(azimuth):
    return -azimuth


def elevation_unconvention(elevation):
    return elevation + 0.5 * np.pi


# cf https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(R.shape[0], dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6



def rotation_matrix_to_azimuth_elevation_depth_tuple(R):
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    assert (isRotationMatrix(R[:3, :3]))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    azimuth = azimuth_unconvention(math.atan2(-R[0, 1], R[0, 0]))
    elevation = elevation_unconvention(math.atan2(-R[1, 2], R[2, 2]))

    depth = - R[2, 3]

    return azimuth, elevation, depth


def rotation_matrix_to_azimuth_elevation_depth(R):
    return np.array(list(rotation_matrix_to_azimuth_elevation_depth_tuple(R)))


def get_euler_angles(T):
    euler = mathutils.Matrix(T).to_euler()
    return euler.x, euler.y, euler.z


def get_euler_angles_degrees(T):
    euler = mathutils.Matrix(T).to_euler()
    euler.order = 'ZYX'
    return euler.x*180.0/np.pi, euler.y*180.0/np.pi, euler.z*180.0/np.pi


def get_euler_angles_degrees_scipy(T):
    from scipy.spatial.transform.rotation import Rotation
    quat = mathutils.Matrix(T).to_quaternion()
    r = Rotation.from_quat(quat)
    return r.as_euler(seq='XYZ', degrees=True)


def T_to_axis_angles(T):
    em = mathutils.Matrix(T).to_quaternion().to_axis_angle()[0]
    return em.x, em.y, em.z


def get_translation(T):
    t = mathutils.Matrix(T).translation
    return t.x, t.y, t.z


def compare_poses(np_pose1, np_pose2):
    """
    :param np_pose1:
    :param np_pose2:
    :return: [angle in rads, translation in metres]
    """
    pose1 = mathutils.Matrix(np_pose1).to_quaternion()
    pose2 = mathutils.Matrix(np_pose2).to_quaternion()
    angle_difference = pose1.rotation_difference(pose2).angle
    translation_difference = np.linalg.norm(np_pose1[:, 3] - np_pose2[:, 3])
    return [angle_difference, np.asscalar(translation_difference)]


def compare_poses_tolerance(np_pose1: np.ndarray, np_pose2: np.ndarray, theta_tol=np.pi/180, t_tol=0.002):
    """
    :param np_pose1: 4x4
    :param np_pose2: 4x4
    :param theta_tol: tolerance value for theta (radians)
    :param t_tol: tolerance value for t (metres)
    :return: (bool: theta within tolerance?, bool: t within tolerance)
    """
    delta_theta, delta_t = compare_poses(np_pose1, np_pose2)
    return np.abs(delta_theta) < theta_tol, np.abs(delta_t) < t_tol


def compare_rotations(np_pose1, np_pose2):
    """
    :param np_pose1:
    :param np_pose2:
    :return: in radians
    """
    pose1 = mathutils.Matrix(np_pose1).to_quaternion()
    pose2 = mathutils.Matrix(np_pose2).to_quaternion()
    return pose1.rotation_difference(pose2).angle


def np_rotation_matrix_to_6d_flat(rotation_matrices):
    """
    takes batch x 3x3 rotation matrix, returns a flat batch x 6 floats for loss
    :param rotation_matrices: batch x (3x3) rotation matrix
    :return: flat batch x 6 floats for loss
    """
    return np.concatenate([rotation_matrices[:, :, 0], rotation_matrices[:, :, 1]], axis=1)


def np_transformation_matrix_to_9d_flat(rotation_matrices):
    """
    takes batch x 4x4 rotation matrix, returns a flat batch x 9 floats for loss - 6 rotation 3 translation
    :param rotation_matrices: batch x (4x4) rotation matrix
    :return: flat batch x 9 floats for loss - 6 rotation 3 translation
    """
    return np.concatenate([rotation_matrices[:, :3, 0], rotation_matrices[:, :3, 1], rotation_matrices[:, :3, 3]], axis=1)


def np_rotation_6d_flat_to_rotation_matrix(rotation_6d_flat):
    batches = rotation_6d_flat.shape[0]
    b1 = rotation_6d_flat[:, :3]
    a2 = rotation_6d_flat[:, 3:]
    return_rotation_matrices_column_1 = b1
    return_rotation_matrices_column_1 /= np.linalg.norm(return_rotation_matrices_column_1, axis=1, keepdims=True)
    return_rotation_matrices_column_2 = a2 - np.sum(b1 * a2, axis=1).reshape((batches, 1)) * b1
    return_rotation_matrices_column_2 /= np.linalg.norm(return_rotation_matrices_column_2, axis=1, keepdims=True)
    return_rotation_matrices_column_3 = np.cross(return_rotation_matrices_column_1,
                                                 return_rotation_matrices_column_2)

    together = np.concatenate((return_rotation_matrices_column_1, return_rotation_matrices_column_2,
                    return_rotation_matrices_column_3), axis=1).reshape((batches, 3, 3))

    return np.swapaxes(together, 1, 2)


def np_rotation_9d_flat_to_transformation_matrix(rotation_9d_flat):
    batches = rotation_9d_flat.shape[0]
    b1 = rotation_9d_flat[:, :3]
    a2 = rotation_9d_flat[:, 3:6]
    return_rotation_matrices_column_1 = b1
    return_rotation_matrices_column_1 /= np.linalg.norm(return_rotation_matrices_column_1, axis=1, keepdims=True)
    return_rotation_matrices_column_2 = a2 - np.sum(b1 * a2, axis=1).reshape((batches, 1)) * b1
    return_rotation_matrices_column_2 /= np.linalg.norm(return_rotation_matrices_column_2, axis=1, keepdims=True)
    return_rotation_matrices_column_3 = np.cross(return_rotation_matrices_column_1,
                                                 return_rotation_matrices_column_2)
    return_rotation_matrices_column_4 = rotation_9d_flat[:, 6:]

    return np.concatenate([
        np.concatenate((return_rotation_matrices_column_1,
                        return_rotation_matrices_column_2,
                        return_rotation_matrices_column_3,
                        return_rotation_matrices_column_4), axis=1).reshape((batches, 4, 3)).swapaxes(2, 1),
                        np.repeat(np.array([[[0, 0, 0, 1]]]), batches, axis=0)], axis=1).reshape((batches, 4, 4))


def rub_to_rdf(T):
    rub_to_rdf_transform = np.eye(4)
    # rub_to_rdf_transform[0, 0] = -1
    rub_to_rdf_transform[1, 1] = -1
    rub_to_rdf_transform[2, 2] = -1
    return np.dot(rub_to_rdf_transform, np.copy(T))


def force_det_one(rotation_matrix):
    # this was added because some matrices output by pnp are determinant -1
    if np.allclose(np.linalg.det(rotation_matrix), -1):
        rotation_matrix[:3, 2] = np.cross(rotation_matrix[:3, 0], rotation_matrix[:3, 1])
    return rotation_matrix
