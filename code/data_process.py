import warnings
warnings.filterwarnings("ignore")
from features import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
import numpy as np


def change_categorical_data_type(df):
    for col in category_lst:
        df[col] = df[col].astype('category')
    return df


def latent_variable(destination_file):
    destinations = pd.read_csv(destination_file)
    pca = PCA(n_components=3)
    dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
    dest_small = pd.DataFrame(dest_small)
    dest_small.columns = ['pca1', 'pca2', 'pca3']
    dest_small["srch_destination_id"] = destinations["srch_destination_id"]
    return dest_small


def delta_to_int(timedelta):
    try:
        return timedelta.days
    except AttributeError:
        return np.nan


def date_to_weekday(date):
    return date.weekday()


def calc_fast_features(df):
    """ given test or training data, return the test_x, test_y"""
    df['dow'] = pd.to_datetime(df['date_time']).apply(date_to_weekday)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['srch_ci'] = pd.to_datetime(df['srch_ci'], errors='coerce')
    df['srch_co'] = pd.to_datetime(df['srch_co'], errors='coerce')
    df['stay_days'] = (df['srch_co'] - df['srch_ci'])
    df['before_days'] = (df['srch_ci'] - df['date_time'])
    df['stay_days'] = df['stay_days'].apply(delta_to_int)
    df['before_days'] = df['before_days'].apply(delta_to_int)
    dest_small = latent_variable(DEST_FILE)
    df1 = df.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest") # todo
    df2 = df1.drop("srch_destination_iddest", axis=1) # todo
    df3 = df2[selected_features]    # select which feature to use
    return df3


def training_processor(train):
    """ given training data, return train_x, train_y """
    train = train.dropna(axis=0, how='any')
    train = change_categorical_data_type(train)

    train_y = train['hotel_cluster']
    train_x = calc_fast_features(train)

    imp = Imputer(strategy='mean', axis=0)
    train_x = imp.fit_transform(train_x)

    return train_x, train_y


def testing_processor(test):
    """ given test data, return test_x, test_y """
    imp = Imputer(strategy='mean', axis=0)
    test_X = calc_fast_features(test)
    test_X = imp.fit_transform(test_X)
    test_Y = test['hotel_cluster']

    return test_X, test_Y
