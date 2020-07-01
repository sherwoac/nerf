import numpy as np
import h5py
import pandas as pd
from LOADER.dataframe_utils import convert_dataframe_dicts_to_columns

__master_group_name = 'dataframe_conversion'
supported_types = ['float', 'int', 'str', 'np.ndarray']

class H5PYWrapper(object):
    @staticmethod
    def create_file_from_dataframe(filename, df):
        _h5py_file = h5py.File(filename, "w")
        column_group_dict = {}
        for column_name in df.columns:
            column_group_dict[column_name] = _h5py_file.create_group(column_name)
            H5PYWrapper.populate_file_from_dataframe(df[column_name], column_group_dict[column_name])
            _h5py_file.flush()

        _h5py_file.close()

    @staticmethod
    def populate_file_from_dataframe(single_col_series, h5_group):
        column_name = single_col_series.name
        # create group per column name
        first_row_of_this_column = single_col_series.head(n=1).iloc[0]
        if isinstance(first_row_of_this_column, float) or (hasattr(first_row_of_this_column, 'shape') and first_row_of_this_column.shape == () and isinstance(first_row_of_this_column, np.float32)):
            try:
                h5_group.create_dataset(column_name + '_data.float', shape=(len(single_col_series),), dtype='f', data=single_col_series.values)
            except:
                print('failed:', column_name, single_col_series.values)
        elif isinstance(first_row_of_this_column, np.ndarray):
            new_shape = ((len(single_col_series),) + first_row_of_this_column.shape)
            h5_group.create_dataset(column_name + '_data.np.ndarray', shape=new_shape, data=np.concatenate(single_col_series.values).reshape(new_shape))

        elif isinstance(first_row_of_this_column, str):
            dt = h5py.special_dtype(vlen=str)
            h5_group.create_dataset(column_name + '_data.str', shape=(len(single_col_series),), dtype=dt, data=single_col_series.values)

        elif isinstance(first_row_of_this_column, bool) or (hasattr(first_row_of_this_column, 'dtype') and first_row_of_this_column.dtype == np.bool):
            h5_group.create_dataset(column_name + '_data.bool', shape=(len(single_col_series),), dtype='u1', data=single_col_series.transform(lambda boolean_value: 1 if boolean_value else 0).astype('int8'))

        elif isinstance(first_row_of_this_column, (int, np.int, np.int64)):
            h5_group.create_dataset(column_name + '_data.int', shape=(len(single_col_series),), data=single_col_series.values.astype(int))

        elif isinstance(first_row_of_this_column, dict):
            col_per_dict_key_df = pd.DataFrame(list(single_col_series))
            dict_group = h5_group.create_group(column_name + '.dict')
            for dict_key_column in col_per_dict_key_df.keys():
                H5PYWrapper.populate_file_from_dataframe(col_per_dict_key_df[dict_key_column], dict_group)
        else:
            raise Exception('unknown type conversion for column: {} from: {}'.format(column_name, type(first_row_of_this_column).__name__))


    @staticmethod
    def populate_dataframe_from_file(h5py_filename):
        h5py_file = h5py.File(h5py_filename, 'r')
        df = pd.DataFrame()
        for key in h5py_file.keys():
            for dataset_name in h5py_file[key]:
                obj = h5py_file[key + '/' + dataset_name]
                if isinstance(obj, h5py.Dataset):
                    df[key] = H5PYWrapper.chunk_values(h5py_file, dataset_name, key)
                elif isinstance(obj, h5py.Group): # recursed dicts
                    dict_of_lists = {}
                    for nested_name in h5py_file[obj.name].keys():
                        actual_array = H5PYWrapper.chunk_values(h5py_file, nested_name, obj.name)
                        if any(type_name in nested_name for type_name in supported_types):
                            nested_name = nested_name[:nested_name.find('.')][:nested_name.find('_data')]
                        dict_of_lists[nested_name] = [array_row for array_row in actual_array]

                    df[key] = pd.DataFrame.from_dict(dict_of_lists).to_dict(orient='records')

                else:
                    raise Exception('unknown h5py type: {}'.format(type(obj)))

        return df

    @staticmethod
    def chunk_values(h5py_file, dataset_name, key):
        # TODO: this bit uses a shed load of memory
        values = h5py_file[key + '/' + dataset_name]
        if '.bool' in dataset_name:

            return np.array(values).astype(bool)
        elif len(values.shape) > 1 and values.shape[0] == len(values):
            actual_array = np.array(values[()])
            return [array_row for array_row in actual_array]

        return values[()]


    @staticmethod
    def get_dataframe_from_file(h5py_filename):
        pass

    @staticmethod
    def print(filename):
        def print_name(name):
            print(name)

        h5_file = h5py.File(filename, "r")
        h5_file.visit(print_name)
