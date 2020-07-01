import pandas as pd
import numpy as np

_special_hdf_indicator_column = '__hdf'


def deep_copy_pandas(input_data):
    import copy
    if isinstance(input_data, pd.DataFrame):
        new_data = pd.DataFrame()
        for column_name in input_data.columns.values:
            new_data[column_name] = deep_copy_pandas(input_data[column_name])
        return new_data

    elif isinstance(input_data, pd.Series):
        new_data = pd.Series()
        for (index, value) in input_data.iteritems():
            new_data.at[index] = deep_copy_pandas(value)
        return new_data

    elif isinstance(input_data, np.ndarray):
        return np.copy(input_data).astype(input_data.dtype)

    else:
        return copy.deepcopy(input_data)


def deep_pandas_equal(input_data1, input_data2, ignore_columns=[]):
    if not type(input_data1) == type(input_data2):
        return False
    if isinstance(input_data1, pd.DataFrame):
        if not all(input_data2.columns.values == input_data1.columns.values):
            return False

        for column_name in input_data1:
            if not column_name in ignore_columns:
                if not deep_pandas_equal(input_data1[column_name].values, input_data2[column_name].values):
                    return False

    elif isinstance(input_data1, np.ndarray):
        if isinstance(input_data1[0], np.ndarray):
            return np.all(np.equal(np.concatenate(input_data1), np.concatenate(input_data2)))
        elif isinstance(input_data1[0], dict):
            for np_row1, np_row2 in zip(input_data1, input_data2):
                for k, v in np_row1.items():
                    if not deep_pandas_equal(np_row1[k], np_row2[k]):
                        return False
        else:
            return np.all(np.equal(input_data1, input_data2))


    else:
        return input_data1 == input_data2

    return True


def convert_bool_columns_to_hdf(df):
    original_column_count = len(df.columns)
    list_of_bool_columns = []
    for column_name in df:
        if df[column_name].dtype == bool or isinstance(df[column_name].head(n=1)[0], bool):
            df[column_name + _special_hdf_indicator_column] = df[column_name].transform(lambda boolean_value: 1 if boolean_value else 0).astype('int8')
            list_of_bool_columns.append(column_name)
    df.drop(list_of_bool_columns, axis=1, inplace=True)
    assert original_column_count == len(df.columns), 'column count changed from: {} to:'.format(original_column_count, len(df.columns))
    return df


def convert_hdf_bool_columns_to_bool(df):
    original_column_count = len(df.columns)
    list_of_hdf_bool_columns = []
    for column_name in df:
        if _special_hdf_indicator_column in column_name:
            bool_column_name = column_name.replace(_special_hdf_indicator_column, '')
            df[bool_column_name] = df[column_name].transform(lambda hdf_boolean_value: True if hdf_boolean_value else False).astype(bool)
            list_of_hdf_bool_columns.append(column_name)
    df.drop(list_of_hdf_bool_columns, axis=1, inplace=True)
    assert original_column_count == len(df.columns), 'column count changed from: {} to:'.format(original_column_count, len(df.columns))
    return df


def convert_dataframe_dicts_to_columns(df):
    list_of_dict_columns = []
    for column_name in df:
        if isinstance(df[column_name].head(n=1)[0], dict):
            list_of_dict_columns.append(column_name)
            for dict_key in df[column_name].head(n=1)[0].keys():
                df[column_name + '.' + dict_key + _special_hdf_indicator_column] = \
                    df[column_name].transform(lambda dict_row: dict_row[dict_key])

    df.drop(list_of_dict_columns, axis=1, inplace=True)
    return df


def convert_dataframe_columns_to_dicts(df):
    dict_columns = {}
    for column_name in df:
        if '.' in column_name and _special_hdf_indicator_column in column_name:
            original_column_name = column_name[0:column_name.find('.')]
            original_dict_key = column_name[column_name.find('.'):].replace(_special_hdf_indicator_column, '')
            dict_columns[original_column_name].append(original_dict_key)

    df[original_column_name] = df[dict_columns[original_column_name]].to_dict('records')
    df.drop(dict_columns[original_column_name], axis=1, inplace=True)
    return df


def report_df_size(df):
    from sys import getsizeof
    overall_size = 0
    for column_name in df:
        value = df[column_name].head(n=1).iloc[0]
        if isinstance(value, np.ndarray):
            size = len(df) * value.nbytes / (1024 ** 2)
            value_type = str(value.dtype)
        else:
            size = df[column_name].apply(getsizeof).sum() / (1024 ** 2)
            value_type = str(type(value))
        overall_size += size
        print('{0:20}: {1:20} MB type: {2}'.format(column_name, size, value_type))

    print('overall: {0:20} MB'.format(overall_size))


def count_test_train(df):
    if 'object_id' in df:
        return df.groupby(['test_train', 'sequence_name', 'object_id']).agg(['count'])
    elif 'object_name' in df:
        return df.groupby(['test_train', 'sequence_name', 'object_name']).agg(['count'])
    else:
        pass


def count_group_by(df, groups_string_list):
        return df.groupby(groups_string_list).agg(['count'])


def count_by_object(df):
    if 'object_id' in df:
        return df.groupby(['test_train', 'object_id']).agg(['count'])
    elif 'object_name' in df:
        return df.groupby(['test_train', 'object_name']).agg(['count'])


def count_truncated_occluded(df):
    if 'object_id' in df:
        return df.groupby(['test_train', 'truncated', 'occluded']).agg(['count'])
    elif 'object_name' in df:
        return df.groupby(['test_train', 'sequence_name', 'object_name']).agg(['count'])
    else:
        return df.groupby(['test_train', 'sequence_name', 'truncated', 'occluded']).agg(['count'])