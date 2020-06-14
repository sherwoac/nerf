import numpy as np


def wrap_image_to_shape(output_shape, wrap_this_image, offset_rc=[0, 0]):
    """
    wrap an rgb ndarray image with a black border like like_this_image
    :param like_this_image:
    :param wrap_this_image:
    :param offset:
    :return: a wrapped image
    """
    assert output_shape[0] >= wrap_this_image.shape[0] + offset_rc[0] \
           and output_shape[1] >= wrap_this_image.shape[1] + offset_rc[1], \
        'incorrect image wrap proposed like: {} wrap: {} offset: {}'.format(output_shape,
                                                                            wrap_this_image.shape,
                                                                            offset_rc)
    output_image = np.zeros(shape=output_shape, dtype=np.uint8)
    output_image[offset_rc[0]:wrap_this_image.shape[0], offset_rc[1]:wrap_this_image.shape[1], :] = np.copy(wrap_this_image)
    return output_image


def tile_images(np_array_of_arrays):
    """

    :param np_array_of_arrays: array of arrays np.array([[image1, image2],[image3, image4]])
    :return: tiled image
    """
    assert len(np_array_of_arrays.shape) > 3 or (np_array_of_arrays.dtype == np.object and len(np_array_of_arrays.shape) == 2), 'illegal shape: np_array_of_arrays: {} '.format(np_array_of_arrays.shape)
    for i, input_array in enumerate(np_array_of_arrays):
        assert isinstance(input_array, np.ndarray), "input array not ndarray: [{}]".format(i)

    channels = input_array.shape[-1]
    row_images = []
    for np_row in range(np_array_of_arrays.shape[0]):
        max_shape_for_this_row = (0, 0)
        for np_col in range(np_array_of_arrays.shape[1]):
            if np_array_of_arrays[np_row, np_col].shape[0] > max_shape_for_this_row[0] and \
                    np_array_of_arrays[np_row, np_col].shape[1] > max_shape_for_this_row[1]:
                max_shape_for_this_row = np_array_of_arrays[np_row, np_col].shape[:2]

        new_cols = max_shape_for_this_row[1] * np_array_of_arrays.shape[1]
        new_rows = max_shape_for_this_row[0]
        row_image = np.zeros(shape=(new_rows, new_cols, channels), dtype=np.uint8)
        for np_col in range(np_array_of_arrays.shape[1]):
            if not np_array_of_arrays[np_row, np_col].shape[:2] == max_shape_for_this_row:
                wrapped_image = wrap_image_to_shape(tuple([*list(max_shape_for_this_row), channels]), np_array_of_arrays[np_row, np_col])
                row_image[:, np_col * max_shape_for_this_row[1]:(np_col + 1) * max_shape_for_this_row[1], :] = wrapped_image
            else:
                row_image[:, np_col * max_shape_for_this_row[1]:(np_col + 1) * max_shape_for_this_row[1], :] = np_array_of_arrays[np_row, np_col]

        row_images.append(row_image)

    max_row_shape = (0, 0)
    for row_image in row_images:
        if row_image.shape[0] >= max_row_shape[0] and row_image.shape[1] >= max_row_shape[1]:
            max_row_shape = row_image.shape[:2]

    output_image = np.zeros(shape=(max_row_shape[0] * len(row_images), max_row_shape[1], 3), dtype=np.uint8)
    for row_number, row_image in enumerate(row_images):
        if not row_image.shape[:2] == max_row_shape:
            wrapped_image = wrap_image_to_shape(tuple([*list(max_row_shape), channels]), row_image)
            output_image[row_number * row_image.shape[0]: (row_number + 1) * wrapped_image.shape[0], :, :] = wrapped_image
        else:
            output_image[row_number * row_image.shape[0]: (row_number + 1) * row_image.shape[0], :, :] = row_image

    return output_image