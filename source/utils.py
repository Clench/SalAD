import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def split_series(series, n_past, n_future):
    '''

    :param series: input time series
    :param n_past: number of past observations
    :param n_future: number of future series
    :return: X, y(label)
    '''
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)

    return X, y


def split_series_33(series_list, n_past, n_future):
    X, y = list(), list()
    for series in series_list:
        series = series.values
        for window_start in range(len(series)):
            past_end = window_start + n_past
            future_end = past_end + n_future
            if future_end > len(series):
                break
            # slicing the past and future parts of the window
            past, future = series[window_start:past_end, :], series[past_end:future_end, :]
            X.append(past)
            y.append(future)

    return X, y



def save(state, epoch, save_dir, name, model):
    with open(save_dir +'/' +name+'_'+ state +".path.tar", "wb") as f:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()}, f)


class MV_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return [self.X, self.y]


def shift_to_categorical(df):
    cols_w_str = list()
    temp = {}
    cnter = 1
    for col in df.columns:
        if df[col].dtype == 'object':
            cols_w_str.append(col)
    for col in cols_w_str:
        for idx in df[col].value_counts().index:
            if col not in temp:
                temp[col] = {}
            temp[col][idx] = cnter
            cnter += 1
        cnter = 1
    df = df.replace(temp)
#     df = df.applymap(temp)
    return df, temp


# Preprocessing includes: 1) Change categorical strings to integer 2) train the scaler with training dataset
def get_train_test_dfs(paths):
    trains = list()
    tests  = list()
    dicts  = list()
    scaler = MinMaxScaler()
    
    for i, path in enumerate(paths):
        temp = pd.read_csv(path, sep=',', header=0)

        # Delete non-variant columns --> Don't do this
        # df = temp.loc[:, (temp != temp.iloc[0]).any()]
        
        # Remove nan columns
#         temp = temp.dropna(axis=1, how='all') # how='all' for checking if everything is nan
        
        # Change categorical values to integers
        df, dict = shift_to_categorical(temp)
        
        if i == 0:
            dicts.append(dict)
        if i < 23:  # Train dataset
            # Fit the scaler with training set
            scaler.partial_fit(df.values)
            trains.append(df)
        else:  # Test dataset
            tests.append(df)
       
    # Scaling
    trains_scaled = list()
    tests_scaled = list()
    for df in trains:
        df[df.columns] = scaler.transform(df)
        trains_scaled.append(df)
    for df in tests:
        df[df.columns] = scaler.transform(df)
        tests_scaled.append(df)

    return trains_scaled, tests_scaled, dicts

# Preprocessing includes: 1) Change categorical strings to integer 2) train the scaler with training dataset
def get_train_test_dfs_px4(path):
    trains = list()
    tests  = list()
    dicts  = list()
    scaler = MinMaxScaler()
    
    
    temp = pd.read_csv(path, sep=',', header=0)

    # Delete non-variant columns
#     temp = temp.loc[:, (temp != temp.iloc[0]).any()]
#     temp = temp[temp.columns[temp.nunique() <= 1]]
    temp = temp.drop(temp.columns[temp.nunique() <= 1], axis = 1)

    # Remove nan columns
    temp = temp.dropna(axis=1, how='all') # how='all' for checking if everything is nan

    # Change categorical values to integers
    temp, dict = shift_to_categorical(temp)

    
    
#     if i < 23:  # Train dataset
#         # Fit the scaler with training set
#         scaler.partial_fit(df.values)
#         trains.append(df)
#     else:  # Test dataset
#         tests.append(df)

    # Scaling
#     trains_scaled = list()
#     tests_scaled = list()
#     for df in trains:
#         df[df.columns] = scaler.transform(df)
#         trains_scaled.append(df)
#     for df in tests:
#         df[df.columns] = scaler.transform(df)
#         tests_scaled.append(df)

#     return trains_scaled, tests_scaled, dicts
    return temp, dict


def get_subsystems(df):
    # Get subsystems
    data_cols = df.columns
    subsystems = list()
    for cols in data_cols:
        subsystems.append(cols.split(':')[0])

    return list(set(subsystems))


def split_subsys(dfs, subsystems):
    df_subsys = {}
    for subsys in subsystems:
        for df in dfs:
            if subsys not in df_subsys:
                df_subsys[subsys] = [df.filter(like=subsys)]
            else:
                df_subsys[subsys].append(df.filter(like=subsys))
    return df_subsys



#### KARI CODES ####
def add_meta(df):
    """
    Feature Engineering for months
    """
    meta      = np.array([each.split(' ')[1:3] for each in df['timestamp']])
    month_arr = meta[:, 0]
    year_arr  = meta[:, 1]

    df['month'] = month_arr
    df['year']  = year_arr
    return df


#### KARI CODES ####
def train_test_split(df, train_month, test_month, train_year, test_year:str, norm=True):
    """
    months: 상반기/하반기
    years : 연도
    
    return: numpy_arr (train/test) and meta (train/test dataframe)
    """ 
    train_month_idx = np.in1d(df['month'].values, train_month)
    train_year_idx = np.in1d(df['year'].values, train_year)
    # train_year_idx = df['year'].values == train_year
    
    test_month_idx = np.in1d(df['month'].values, test_month)
    test_year_idx = np.in1d(df['year'].values, test_year)
    # test_year_idx = df['year'].values == test_year
    
    train_idx = np.logical_and(train_month_idx, train_year_idx)
    test_idx = np.logical_and(test_month_idx, test_year_idx)
    
    train_df = df[train_idx]
    test_df  = df[test_idx]
    
    if norm:
        s = MinMaxScaler()
        train = s.fit_transform(train_df.drop(['timestamp', 'month', 'year'], axis=1).values)
        test  = s.transform(test_df.drop(['timestamp', 'month', 'year'], axis=1).values)
        return train, test, s, [train_df, test_df]
    
    else:
        train = train_df.values
        test = test_df.values
        return train, test, [train_df, test_df]


# def junkfunc():
#     label_path = '../dataset/dji_px4/dataset/dji/dji_normal.csv'
#
#     label_data = pd.read_csv(label_path, sep=',', header=0)
#
#     # IF the data contains multiple csvs
#     paths = glob.glob('../dataset/temp/*.csv')
#
#     i = 0
#     data = list()  # initialization
#     for path in paths:
#         print("path:", path)
#         if i == 0:
#             data = pd.read_csv(path, sep=',', header=0)
#             print('len: ', len(data))
#             i += 1
#         else:
#             temp = pd.read_csv(path, sep=',', header=0)
#             print('len: ', len(temp))
#             data = data.append(temp)
#
#     data.head()
#


