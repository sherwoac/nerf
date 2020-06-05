import struct
import tensorflow as tf


def numpy_repeat_equiv(matrix_to_repeat, repeats):
    matrix_shape = tf.shape(matrix_to_repeat)[-2:]
    return tf.reshape(tf.tile(tf.reshape(matrix_to_repeat, shape=[-1]), [repeats]), shape=(repeats, matrix_shape[-2], matrix_shape[-1]))


def tf_summary_string_to_dict(summary_str):
    idx = 0
    ret_dict = {}
    while idx < len(summary_str):
        item_len = struct.unpack_from('B', summary_str, idx + 1)[0]
        name_len = struct.unpack_from('B', summary_str, idx + 3)[0]
        name = str(summary_str[idx + 4:idx + 4 + name_len], "utf-8")
        value = struct.unpack_from('<f', summary_str, idx + 5 + name_len)[0]
        ret_dict[name] = value
        idx += item_len + 2
    return ret_dict


def allclose(x, y, rtol=1e-5, atol=1e-8):
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)
