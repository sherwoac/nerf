import tensorflow as tf
from MODELS.tf_tools import allclose


def rotation_matrix_to_label_6d_flat(rotation_matrices):
    """
    takes batch x 3x3 rotation matrix, returns a flat batch x 6 floats for loss
    :param rotation_matrices: batch x (3x3) rotation matrix
    :return: flat batch x 6 floats for loss
    """
    return tf.concat([rotation_matrices[:, :, 0], rotation_matrices[:, :, 1]], axis=1)


def label_6d_flat_to_rotation_matrix(rotation_6d_flat):
    """
    opposite of above, converts 6d flat rotation representation into 3x3 rotation matrix
    :param rotation_6d_flat: batch x 6d flat
    :return: new batch x (3x3) rotation matrix
    """
    batches = tf.shape(rotation_6d_flat)[0]

    return_rotation_matrices_column_1 = rotation_6d_flat[:, :3]
    return_rotation_matrices_column_1 = return_rotation_matrices_column_1 / tf.linalg.norm(return_rotation_matrices_column_1, axis=1, keepdims=True)

    b1 = return_rotation_matrices_column_1
    a2 = rotation_6d_flat[:, 3:]

    return_rotation_matrices_column_2 = a2 - tf.expand_dims(tf.reduce_sum(tf.multiply(b1, a2), axis=1), 1) * b1
    return_rotation_matrices_column_2 = return_rotation_matrices_column_2 / tf.linalg.norm(return_rotation_matrices_column_2, axis=1, keepdims=True)

    return_rotation_matrices_column_3 = tf.linalg.cross(return_rotation_matrices_column_1, return_rotation_matrices_column_2)

    matrix = tf.reshape(tf.concat([return_rotation_matrices_column_1, return_rotation_matrices_column_2, return_rotation_matrices_column_3], axis=-1), (batches, 3, 3))
    return tf.transpose(matrix, [0, 2, 1])


def transformation_matrix_to_label_9d_flat(transformation_matrices):
    """

    :param transformation_matrices:
    :return:
    """
    return tf.concat([transformation_matrices[:, :3, 0], transformation_matrices[:, :3, 1], transformation_matrices[:, :3, 3]], axis=1)


def label_9d_flat_and_t_to_T(rotation_and_t_9d_flat):
    """
    take a 9 point label and convert it into a set of
    :param rotation_and_t_9d_flat: batch x 9d
    :return: batch x (4x4) T
    """
    batches = tf.shape(rotation_and_t_9d_flat)[0]

    return_rotation_matrices_column_1 = rotation_and_t_9d_flat[:, :3]
    return_rotation_matrices_column_1 = return_rotation_matrices_column_1 / tf.linalg.norm(return_rotation_matrices_column_1, axis=1, keepdims=True)

    b1 = return_rotation_matrices_column_1
    a2 = rotation_and_t_9d_flat[:, 3:6]

    return_rotation_matrices_column_2 = a2 - tf.expand_dims(tf.reduce_sum(tf.multiply(b1, a2), axis=1), 1) * b1
    return_rotation_matrices_column_2 = return_rotation_matrices_column_2 / tf.linalg.norm(return_rotation_matrices_column_2, axis=1, keepdims=True)

    # # check column 1 and 2 orthogonal
    # column1_column2_orthogonal_assert = tf.Assert(
    #     allclose(
    #         tf.linalg.tensordot(return_rotation_matrices_column_1, return_rotation_matrices_column_2, 2),
    #         tf.cast(0.0, dtype=return_rotation_matrices_column_1.dtype),
    #         rtol=1e-1
    #         ),
    #     ["column1_column2_orthogonal_assert:", return_rotation_matrices_column_1, return_rotation_matrices_column_2])

    return_rotation_matrices_column_3 = tf.linalg.cross(return_rotation_matrices_column_1, return_rotation_matrices_column_2)

    # check determinant of R
    Rs = tf.reshape(
        tf.concat(
            (return_rotation_matrices_column_1,
             return_rotation_matrices_column_2,
             return_rotation_matrices_column_3), axis=1),
        (batches, 3, 3))

    det_Rs = tf.linalg.det(Rs)
    determinant_of_R_is_one = tf.Assert(
        allclose(
            det_Rs,
            tf.cast(1.0, dtype=det_Rs.dtype)),
        ["determinant_of_R_is_one:", det_Rs])

    return_rotation_matrices_column_4 = rotation_and_t_9d_flat[:, 6:]

    matrix = tf.concat([
        tf.transpose(
            tf.reshape(
                tf.concat((return_rotation_matrices_column_1,
                           return_rotation_matrices_column_2,
                           return_rotation_matrices_column_3,
                           return_rotation_matrices_column_4), axis=1),
                (batches, 4, 3)),
            [0, 2, 1]),
        tf.tile(tf.constant([[[0.0, 0.0, 0.0, 1.0]]], dtype=return_rotation_matrices_column_1.dtype), [batches, 1, 1])], axis=1)

    # with tf.control_dependencies([column1_column2_orthogonal_assert, determinant_of_R_is_one]):
    return matrix


def test_6d_flat():
    import numpy as np
    import mathutils
    from MODELS.rotations import np_rotation_6d_flat_to_rotation_matrix, np_rotation_matrix_to_6d_flat
    batches = 2
    original_matrix = mathutils.Matrix.Rotation(np.pi/4, 3, "X")
    batch_matrices = np.broadcast_to(original_matrix, (batches, 3, 3))
    np_batch_matrices_flat = np_rotation_matrix_to_6d_flat(batch_matrices)
    np_batch_matrices = np_rotation_6d_flat_to_rotation_matrix(np_batch_matrices_flat)
    assert np.allclose(batch_matrices, np_batch_matrices)

    # now TF version
    sess = tf.Session()
    with sess.as_default():
        tf_I_batch = tf.convert_to_tensor(batch_matrices, dtype=tf.float32)
        tf_flat_6d = rotation_matrix_to_label_6d_flat(tf_I_batch).eval()
        tf_I_batch_return = label_6d_flat_to_rotation_matrix(tf_flat_6d).eval()
        assert np.all(tf_I_batch_return == batch_matrices), f"failed: {batch_matrices} not {tf_I_batch_return}."


def test_9d_flat():
    import numpy as np
    import mathutils
    from MODELS.rotations import np_rotation_9d_flat_to_transformation_matrix, np_transformation_matrix_to_9d_flat
    batches = 2
    original_matrix = mathutils.Matrix.Rotation(np.pi/4, 4, "X")
    original_matrix.translation = mathutils.Vector([1,2,3])
    batch_matrices = np.broadcast_to(original_matrix, (batches, 4, 4))
    np_batch_matrices_flat = np_transformation_matrix_to_9d_flat(batch_matrices)
    np_batch_matrices = np_rotation_9d_flat_to_transformation_matrix(np_batch_matrices_flat)
    assert np.allclose(batch_matrices, np_batch_matrices), f"failed: {batch_matrices} not {np_batch_matrices}."

    # now TF version
    sess = tf.Session()
    with sess.as_default():
        tf_I_batch = tf.convert_to_tensor(batch_matrices, dtype=tf.float32)
        tf_flat_9d = transformation_matrix_to_label_9d_flat(tf_I_batch).eval()
        tf_I_batch_return = label_9d_flat_and_t_to_T(tf_flat_9d).eval()
        assert np.allclose(tf_I_batch_return, batch_matrices), f"failed: {batch_matrices} not {tf_I_batch_return}."


if __name__ == '__main__':
    tf.enable_eager_execution()
    test_6d_flat()
    test_9d_flat()
