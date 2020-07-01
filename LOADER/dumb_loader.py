import os

import math
from datetime import datetime
import numpy as np
import pandas as pd
import glob
from LOADER.h5py_wrapper import H5PYWrapper

_hdf_filename_postfix = '.h5'
_pkl_filename_postfix = '.pkl'


def get_df_info(filename, df):
    loaded_text = _get_file_modified_time(filename)
    print('file: {} modified: {} len: {}'.format(filename, loaded_text, len(df)))


def open_dataframe(filename):
    df = open_dataframe_silent(filename)
    get_df_info(filename, df)
    return df


def open_dataframe_silent(filename):
    if _hdf_filename_postfix in filename:
        return _open_dataframe_hdf(filename)
    elif _pkl_filename_postfix in filename:
        return _open_dataframe_pkl(filename)
    else:
        assert True, 'unknown filetype in: {}'.format(filename)


def save_dataframe(filename, dataframe):
    filename = os.path.expanduser(filename)
    if _hdf_filename_postfix in filename:
        save_dataframe_hdf(filename, dataframe)
    elif _pkl_filename_postfix in filename:
        save_dataframe_pkl(dataframe, filename)
    else:
        assert True, 'unknown filetype in: {}'.format(filename)


def open_dataframe_pkl(filename):
    loaded_text = _get_file_modified_time(filename)
    print(loaded_text)
    return _open_dataframe_pkl(filename) 


def open_dataframe_pkl_silent(filename):
    return _open_dataframe_pkl(filename)


def _open_dataframe_hdf(filename):
    return H5PYWrapper.populate_dataframe_from_file(filename)


def _get_file_modified_time(filename):
    assert os.path.exists(filename), "dataframe file not found at: {}".format(filename)
    return datetime.fromtimestamp(os.path.getmtime(filename))


def _open_dataframe_pkl(filename):
    assert os.path.exists(filename), "pkl file not found at: {}".format(filename)
    return pd.read_pickle(filename)


def save_dataframe_pkl(df, write_out_filename):
    pd.to_pickle(df, filepath_or_buffer=write_out_filename)
    assert os.path.isfile(write_out_filename), "saved file not found: {}".format(write_out_filename)


def save_dataframe_hdf(filename, df):
    H5PYWrapper.create_file_from_dataframe(filename, df)
    assert os.path.isfile(filename), "saved file not found: {}".format(filename)


def save_dataframe_hdf_chunked(filename, df):
    chunk_size = 1000
    number_of_splits = math.ceil(len(df) / chunk_size)
    for index, df_split_input in enumerate(np.array_split(df, number_of_splits)):
        try:
            if len(df_split_input) > 0:
                output_filename = filename.replace('.h5', '') + '_' + str(index).zfill(3) + '.h5'
                df_split_input.to_hdf(output_filename, 'df', mode='w', format='fixed')
                assert os.path.exists(output_filename), 'created file not found: {}'.format(output_filename)
        except Exception as e:
            print(e)
            print(df.info())


def open_dataframe_hdf_chunked(read_in_directory):
    filename_pattern = read_in_directory + '/*'
    full_filenames = [filename for filename in sorted(glob.glob(filename_pattern))]
    assert len(full_filenames) > 0, 'no h5 split files found according to pattern: {}'.format(filename_pattern)
    dfs = []
    for filename in full_filenames:
        dfs.append(_open_dataframe_hdf(filename))

    df = pd.concat(dfs, axis=0)
    df.reset_index(drop=True, inplace=True)
    return df



