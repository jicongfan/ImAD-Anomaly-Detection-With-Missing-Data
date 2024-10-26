__author__ = 'XF'
__date__ = '2023/09/01'

'''
    data preprocessing.
'''
import os
from os import path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import numpy as np
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scanpy import read_h5ad

from configs import DATA_DIR



def Adult_preprocess():

    # https://archive.ics.uci.edu/dataset/2/adult

    data_path = osp.join(DATA_DIR, 'Adult/adult.data')

    # read data
    df_data = pd.read_csv(data_path, header=None, names=['age', 'workclass', 'fnlwgt', 'education',
                                                                    'education-num', 'marital-status', 'occupation',
                                                                    'relationship', 'race', 'sex', 'capital-gain',
                                                                    'capital-loss',
                                                                    'hours-per-week', 'native-country', 'y'])
    # data_path = osp.join(DATA_DIR, 'Adult/adult.test')
    # df_data_test = pd.read_csv(data_path, header=None, names=['age', 'workclass', 'fnlwgt', 'education',
    #                                                                 'education-num', 'marital-status', 'occupation',
    #                                                                 'relationship', 'race', 'sex', 'capital-gain',
    #                                                                 'capital-loss',
    #                                                                 'hours-per-week', 'native-country', 'y'])
    # # data info
    # print(df_data.info()) # (32561, 15)
    # print(df_data_test.info()) # (16281, 15)

    # # all the elements of the first row in adutl.test are NaN.
    # # print(df_data_test.iloc[0, :])
    # df_data = pd.concat([df_data, df_data_test[1:]])
    print(f'all data: {len(df_data.iloc[:, 0])}')

    # data cleaning
    df_data_clean = None
    for i in range(len(df_data.iloc[:, 0])):
        missing_value = False
        for j in range(len(df_data.iloc[0, :])):
            if str(df_data.iloc[i, j]) == ' ?':
                missing_value = True
                break  
        if not missing_value:
            if df_data_clean is None:
                df_data_clean = df_data.iloc[i:i+1]
            else:
                df_data_clean = pd.concat([df_data_clean, df_data.iloc[i:i+1]])

    print(f'clean data: {df_data_clean.shape}')

    # df_data_clean.loc[df_data_clean['y'] == ' >50K.', 'y'] = 1
    # df_data_clean.loc[df_data_clean['y'] == ' <=50K.', 'y'] = 0
    df_data_clean.loc[df_data_clean['y'] == ' >50K', 'y'] = 1
    df_data_clean.loc[df_data_clean['y'] == ' <=50K', 'y'] = 0
    df_data_clean.loc[df_data_clean['sex'] == ' Female', 'sex'] = 1
    df_data_clean.loc[df_data_clean['sex'] == ' Male', 'sex'] = -1

    # normal_data = df_data_clean.loc[df_data_clean['y'] == 0]
    # abnormal_data = df_data_clean.loc[df_data_clean['y'] == 1]
    # print(f'normal data: {normal_data.shape}')
    # print(f'abnormal data: {abnormal_data.shape}')



    print(f'a instance before preprocess: {df_data_clean.values[0]}')
    for i in ['workclass', 'marital-status', 'occupation', 'education', 'relationship', 'race', 'native-country']:
        column_value = list(set(df_data_clean[i].values))
        # print(len(column_value), column_value)

        for j, value in enumerate(column_value):
            df_data_clean.loc[df_data_clean[i] == value, i] = j + 1
    
    df_data_clean = df_data_clean[['sex', 'age', 'native-country', 'race', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                       'relationship', 'capital-gain', 'capital-loss',
                       'hours-per-week', 'y']]
    print(f'a instance after preprocess: {df_data_clean.values[0]}')
    
    # saving complete data
    complete_data = df_data_clean.iloc[:, :].values.tolist()
    complete_data = np.array(complete_data, dtype=np.float32)

    
    print(f'complete_data: {type(complete_data), complete_data.shape}')
    # save_path = osp.join(DATA_DIR, 'Adult/complete_data.npy')
    # np.save(save_path, complete_data)
    # print(f'Saving complete data successfully!')


def ODDs_preprocess(data_name):

    # http://odds.cs.stonybrook.edu/arrhythmia-dataset/
    # http://odds.cs.stonybrook.edu/thyroid-dise

    if data_name == 'arrhythmia':
        data_dir = osp.join(DATA_DIR, 'Arrhythmia')
        data_path = osp.join(data_dir, 'arrhythmia.mat')
    elif data_name == 'speech':
        data_dir = osp.join(DATA_DIR, 'Speech')
        data_path = osp.join(data_dir, 'speech.mat')

    dataset = loadmat(data_path)

    data = np.concatenate((dataset['X'], dataset['y']), axis=1)
    print(f'data shape: {data.shape}')

    # saving complete data
    # np.save(osp.join(data_dir, 'complete_data.npy'), data)
    print('Saving complete data successfully!')


def get_missing_data(data_path, missing_rate, save_dir):

    print(f'missing rate: {missing_rate * 100}%')

    complete_data = np.load(data_path, allow_pickle=True)
    mask = np.random.rand(*complete_data[:, :-1].shape) < missing_rate # True for missing values, false for others
    miss_data = np.copy(complete_data)

    miss_data[:, :-1][mask] = np.nan

    # split train and test set
    normal_data = miss_data[miss_data[:,-1] == 0][:, :-1]
    abnormal_data = miss_data[miss_data[:, -1] == 1][:, :-1]

    assert len(normal_data) > len(abnormal_data)

    num_train_data = len(normal_data) - len(abnormal_data)
    train_data = normal_data[:num_train_data]
    train_lab = np.zeros(len(train_data))
    test_normal_data = normal_data[num_train_data:]
    test_data = np.concatenate((test_normal_data, abnormal_data))
    test_lab = np.concatenate((np.zeros(len(test_normal_data)), np.ones(len(abnormal_data))))

    print(f'train data: {train_data.shape}')
    print(f'test data: {test_data.shape}')

    # normalization
    mu = np.nanmean(train_data, axis=0)
    std = np.nanstd(train_data, axis=0)
    std[std == 0] = 1
    print(f'mu: {mu.shape}')
    print(f'std: {std.shape}')
    train_data = (train_data - mu) / std
    test_data = (test_data - mu) / std

    # mean imputation
    train_nan = np.isnan(train_data)
    train_data[train_nan] = 0.0
    test_nan = np.isnan(test_data)
    test_data[test_nan] = 0.0

    # saving data

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    np.save(osp.join(save_dir, 'train_data.npy'), train_data)
    np.save(osp.join(save_dir, 'train_labels.npy'), train_lab)
    np.save(osp.join(save_dir, 'test_data.npy'), test_data)
    np.save(osp.join(save_dir, 'test_labels.npy'), test_lab)
    print('Saving mean-imputed data successfully!')


def KDD_preprocess(num_normal=0, num_abnormal=0):

    path = osp.join(DATA_DIR, 'KDD')
    file_names = [osp.join(path, "kddcup.data_10_percent.gz"), osp.join(path, "kddcup.names")]

    column_name = pd.read_csv(file_names[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
    column_name.loc[column_name.shape[0]] = ['status', ' symbolic.']
    data = pd.read_csv(file_names[0], header=None, names=column_name['f_names'].values)
    data_symbolic = column_name[column_name['f_types'].str.contains('symbolic.')]
    data_continuous = column_name[column_name['f_types'].str.contains('continuous.')]
    samples = pd.get_dummies(data.iloc[:, :-1], columns=data_symbolic['f_names'][:-1])

    sample_keys = samples.keys()
    continuous_idx = []
    for cont_idx in data_continuous['f_names']:
        continuous_idx.append(sample_keys.get_loc(cont_idx))

    labels = np.reshape(np.where(data['status'] == 'normal.', 1, 0), (-1, 1))
    
    if num_normal > 0 and num_abnormal > 0:
        normal_data = samples[labels == 0]
        abnormal_data = samples[labels == 1]
        normal_data = np.random.permutation(normal_data)
        abnormal_data = np.random.permutation(abnormal_data)
        used_normal_data = normal_data[:num_normal, :]
        used_abnormal_data = abnormal_data[:num_abnormal, :]
        samples = np.concatenate((used_normal_data, used_abnormal_data))
        labels = np.reshape(np.concatenate((np.zeros(len(used_normal_data)), np.ones(len(used_abnormal_data)))), (-1, 1))
        print(f'#================ data info ===============#')
        print(f'normal data: {used_normal_data.shape}')
        print(f'abnormal data: {used_abnormal_data.shape}')
        print(f'#================ data info ===============#')
        complete_data = np.concatenate((samples, labels), axis=1)
        print(f'data: {type(samples), samples.shape}')
        print(f'labels: {type(labels), labels.shape}')
        print(f'complete data: {complete_data.shape}')
        # saving complete data
        np.save(osp.join(path, f'complete_data_{num_normal}_{num_abnormal}.npy'), complete_data)
        print('Saving complete data successfully!')

    else :

        complete_data = np.concatenate((samples, labels), axis=1)
        print(f'data: {type(samples), samples.shape}')
        print(f'labels: {type(labels), labels.shape}')
        print(f'complete data: {complete_data.shape}')

        # saving complete data
        np.save(osp.join(path, 'complete_data.npy'), complete_data)
        print('Saving complete data successfully!')


def KDD_get_missing_data(data_path, missing_rate, continuous_idx=None, save_path=None):

    print(f'Missing rate: {missing_rate * 100}%')

    complete_data = np.load(data_path, allow_pickle=True)
    mask = np.random.rand(*complete_data[:, :-1].shape) < missing_rate # True for missing values, false for others
    miss_data = np.copy(complete_data)

    miss_data[:, :-1][mask] = np.nan

    # split train and test set
    normal_data = miss_data[miss_data[:,-1] == 0][:, :-1]
    abnormal_data = miss_data[miss_data[:, -1] == 1][:, :-1]

    assert len(normal_data) > len(abnormal_data)

    num_train_data = len(normal_data) - len(abnormal_data)
    train_data = normal_data[:num_train_data]
    train_lab = np.zeros(len(train_data))
    test_normal_data = normal_data[num_train_data:]
    test_data = np.concatenate((test_normal_data, abnormal_data))
    test_lab = np.concatenate((np.zeros(len(test_normal_data)), np.ones(len(abnormal_data))))

    print(f'train data: {train_data.shape}')
    print(f'test data: {test_data.shape}')

    # normalization

    train_data, test_data = norm_kdd_data(train_data, test_data, continuous_idx=continuous_idx)

    # mean imputation
    train_nan = np.isnan(train_data)
    train_data[train_nan] = 0.0
    test_nan = np.isnan(test_data)
    test_data[test_nan] = 0.0
    # print(np.sum(1 * train_nan))
    # print(np.sum(1 * test_nan))
    # print(np.sum(1 * np.isnan(train_data)))
    # print(np.sum(1 * np.isnan(test_data)))

    # saving data
    print('saving data...')
    if not osp.exists(save_path):
        os.makedirs(save_path)

    np.save(osp.join(save_path, 'train_data.npy'), train_data)
    np.save(osp.join(save_path, 'train_labels.npy'), train_lab)
    np.save(osp.join(save_path, 'test_data.npy'), test_data)
    np.save(osp.join(save_path, 'test_labels.npy'), test_lab)


def norm_kdd_data(train_data, test_data, continuous_idx):

    symbolic_idx = np.delete(np.arange(train_data.shape[1]), continuous_idx)
    mu = np.nanmean(train_data[:, continuous_idx], 0, keepdims=True)
    std = np.nanstd(train_data[:, continuous_idx], 0, keepdims=True)
    std[std == 0] = 1

    train_continual = (train_data[:, continuous_idx] - mu) / std
    train_normalized = np.concatenate([train_data[:, symbolic_idx], train_continual], 1)
    test_continual = (test_data[:, continuous_idx] - mu) / std
    test_normalized = np.concatenate([test_data[:, symbolic_idx], test_continual], 1)

    return train_normalized, test_normalized


def split_train_test_data(dataset, label_normal=0, label_abnormal=1):

    data_path = osp.join(DATA_DIR, f'{dataset}/complete_data.npy')

    data = np.load(data_path, allow_pickle=True)

    normal_data = data[data[:, -1] == label_normal]
    abnormal_data = data[data[:, -1] == label_abnormal]

    print(f'normal data: {normal_data.shape}')
    print(f'abnormal data: {abnormal_data.shape}')
    

    assert len(normal_data) > len(abnormal_data)

    num_train_data = len(normal_data) - len(abnormal_data)

    train_data = normal_data[:num_train_data][:, :-1]
    train_lab = np.zeros(len(train_data))

    test_normal_data = normal_data[num_train_data:][:, :-1]
    test_data = np.concatenate((test_normal_data, abnormal_data[:, :-1]))
    test_lab = np.concatenate((np.zeros(len(test_normal_data)), np.ones(len(abnormal_data))))

    print(f'train data: {train_data.shape}')
    print(f'test data: {test_data.shape}')


    # saving data
    save_path = osp.join(DATA_DIR, f'{dataset}/processed')
    np.save(osp.join(save_path, 'train_data.npy'), train_data)
    np.save(osp.join(save_path, 'train_labels.npy'), train_lab)
    np.save(osp.join(save_path, 'test_data.npy'), test_data)
    np.save(osp.join(save_path, 'test_labels.npy'), test_lab)
    print('saving train, test data successfully!')


def Bio_data(dataset):
    
    print(f'Preprocess [{dataset}] ========================')

    if dataset == 'Usoskin':
        data_dir = osp.join(DATA_DIR, 'Usoskin')
        data_path = osp.join(data_dir, 'usoskin_x.csv')
        label_path = osp.join(data_dir, 'usoskin_label.csv')
        data = load_CSV(data_path, label_path)
    elif dataset == 'Bladder':
        data_dir = osp.join(DATA_DIR, 'Bladder')
        data_path = osp.join(data_dir, 'Quake_10x_Bladder.h5ad')
        data = load_h5ad(data_path, normal_class=0, abnormal_class=3)
    elif dataset == 'Seq2_Heart':
        data_dir = osp.join(DATA_DIR, 'Seq2_Heart')
        data_path = osp.join(data_dir, 'Quake_Smart-seq2_Heart.h5ad')
        data = load_h5ad(data_path, normal_class=4, abnormal_class=6)
    elif dataset == 'Segerstolpe':
        data_dir = osp.join(DATA_DIR, 'Segerstolpe')
        data_path = osp.join(data_dir, 'Seg_Pancreas_x.csv')
        label_path = osp.join(data_dir, 'Seg_Pancreas_label.csv')
        data = load_CSV(data_path, label_path)


    if dataset in ['Bladder', 'Seq2_Heart']:
        print(f'before cleaning: { data.shape}')
        # data[:,:-1][data[:,:-1] == 0.0] = np.nan

    print(f'data shape: {data.shape}')

    # saving complete data
    np.save(osp.join(data_dir, 'complete_data.npy'), data)
    print('Saving complete data successfully!')
          

def load_h5ad(file_path, normal_class=None, abnormal_class=None):

    annData = read_h5ad(file_path)

    X=pd.DataFrame(annData.X.todense())

    cell_name=annData.obs.index
    print(f'Cell Name: {len(cell_name)}')
    chr_name=annData.var.index
    print(f'Chr Name: {len(chr_name)}')
    X.index=cell_name
    X.columns=chr_name

    x=X.values
    print(f'X: {type(x)}, {x.shape}')

    y = np.array(annData.obs.cell_type1.values)
    label_attr = np.unique(y)
    print(f'Class type: {label_attr}')

    for i in range(len(label_attr)):
        idx=np.where(y==label_attr[i])
        y[idx]=i
        print(f'{label_attr[i]}: {idx[0].shape}')
    y = np.array(y).astype('int').reshape(-1, 1)
    print(f'y: {y.shape}')

    complete_data  = np.concatenate((x, y), axis=1)
    if normal_class is not None and abnormal_class is not None:
        normal_data = complete_data[complete_data[:, -1] == normal_class]
        abnormal_data = complete_data[complete_data[:, -1] == abnormal_class]
        normal_data[normal_data[:, -1] == normal_class, -1] = 0
        abnormal_data[abnormal_data[:, -1] == abnormal_class, -1] = 1
        complete_data = np.concatenate((normal_data, abnormal_data), axis=0)

    return complete_data


def load_CSV(data_path, label_path):

    data = pd.read_csv(data_path, index_col=0)

    print(data.info)
    x = np.transpose(data.values)
    print(f'X: {x.shape}')

    label = pd.read_csv(label_path, index_col=0)

    label_attr = np.unique(label.values)
    y = label.values
    print(f'Class type: {label_attr}')
    for i in range(len(label_attr)):
        idx = np.where(y == label_attr[i])
        y[idx] = i
        print(f'{label_attr[i]}: {idx[0].shape}')
    y = np.array(y).astype('int').reshape(-1, 1)

    print(f'y: {y.shape}')

    data = np.concatenate((x, y), axis=1)

    return data


def Iot_data(dataset):

    if dataset == 'Botnet':
        data_dir = osp.join(DATA_DIR, 'Botnet')
        normal_data_path = osp.join(data_dir, 'benign_traffic.csv')
        abnormal_data_path_scan = osp.join(data_dir, 'scan.csv')
        abnormal_data_path_combo = osp.join(data_dir, 'combo.csv')
        abnormal_data_path_junk = osp.join(data_dir, 'junk.csv')
        abnormal_data_path_tcp = osp.join(data_dir, 'tcp.csv')
        abnormal_data_path_udp = osp.join(data_dir, 'udp.csv')
    else:
        raise Exception(f'Unknown dataset [{dataset}]!')

    normal_data = pd.read_csv(normal_data_path, header=0)
    scan_data = pd.read_csv(abnormal_data_path_scan, header=0)
    combo_data = pd.read_csv(abnormal_data_path_combo, header=0)
    junk_data = pd.read_csv(abnormal_data_path_junk, header=0)
    tcp_data = pd.read_csv(abnormal_data_path_tcp, header=0)
    udp_data = pd.read_csv(abnormal_data_path_udp, header=0)

    # print('normal data ============================')
    # print(normal_data.info)
    # print('scan data ===========================')
    # print(scan_data.info)
    normal_data = normal_data.values
    normal_data = np.concatenate((normal_data, np.zeros(normal_data.shape[0]).reshape(-1, 1)), axis=1)

    scan_data = scan_data.values[:1000, :]
    combo_data = combo_data.values[:1000, :]
    junk_data = junk_data.values[:1000, :]
    tcp_data = tcp_data.values[:1000, :]
    udp_data = udp_data.values[:1000, :]

    
    abnormal_data = np.concatenate((scan_data, combo_data, junk_data, tcp_data, udp_data))
    abnormal_data = np.concatenate((abnormal_data, np.ones(abnormal_data.shape[0]).reshape(-1, 1)), axis=1)
    print(f'normal data: {normal_data.shape}')
    print(f'abnormal data: {abnormal_data.shape}')
    data = np.concatenate((normal_data, abnormal_data))
    print(f'data: {data.shape}')

    # saving
    np.save(osp.join(data_dir, 'complete_data.npy'), data)
    print('Saving complete data successfully!')


def Titanic_process():

    train_data_path = osp.join(DATA_DIR, 'Titanic/train.csv')
    df_row = pd.read_csv(train_data_path)
    print(f'row data size: {df_row.shape}')

    # The features 'ticket' and 'cabin' have many missing values and so can't add much value to our analysis. 
    # and the 'PassengerId' and 'Name' are not useful for anomaly detection.
    df_row = df_row.drop(['PassengerId', 'Name'], axis=1)

    df = df_row[['Sex', 'Ticket', 'Cabin', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Survived']]

    df.loc[df['Sex'] == 'female', 'Sex'] = -1
    df.loc[df['Sex'] == 'male', 'Sex'] = 1


    for i in ['Embarked', 'Ticket', 'Cabin']:
        column_value = list(set(df[i].values))
        # print(len(column_value), column_value)

        for j, value in enumerate(column_value):
            df.loc[df[i] == value, i] = j + 1

    data = df.values
    for i, sample in enumerate(data):
        for j, entry in enumerate(sample):
            if np.isnan(entry):
                data[i, j] = np.nan

    normal_data = data[data[:, -1] == 0]
    abnormal_data = data[data[:, -1] ==  1][:100]
    print(f'normal data: {normal_data.shape}')
    print(f'a normal sample: {normal_data[0]}')
    print(f'abnormal data: {abnormal_data.shape}')
    print(f'a abnormal sample: {abnormal_data[0]}')

    complete_data = np.concatenate((normal_data, abnormal_data), axis=0)

    print(f'complete data: {complete_data.shape}')
    # np.save(osp.join(DATA_DIR, 'Titanic/complete_data.npy'), complete_data)
    # print('Saving complete data successfully!')


def MovieLens_preprocess():

    movie_path = osp.join(DATA_DIR, 'MovieLens1M/movies.dat')
    rating_path = osp.join(DATA_DIR, 'MovieLens1M/ratings.dat')
    users_path = osp.join(DATA_DIR, 'MovieLens1M/users.dat')

    movie_data = pd.read_csv(movie_path, sep='::', header=None, encoding='ISO-8859-1',  engine='python')
    rating_data = pd.read_csv(rating_path, sep='::', header=None, encoding='ISO-8859-1', engine='python')
    users_data = pd.read_csv(users_path, sep='::', header=None, encoding='ISO-8859-1', engine='python')

    ages = set(users_data.values[:, 2])
    age_dict = {}
    for age in list(ages):
        age_dict.setdefault(str(age), 0)

    for row in users_data.values:
        age_dict[str(row[2])] += 1

    # for key, value in age_dict.items():

    #     print(f'age-{key}: {value, (value / users_data.shape[0])}')

    print(f'movie data: {movie_data.shape}')
    print(f'rating data: {rating_data.shape}')


    data = [[0 for _ in range(movie_data.shape[0])] for _ in range(len(set(rating_data.values[:,0])))]
    data = np.array(data)

    print(f'data: {data.shape}')

    user = 0
    num_missing_sample = 0
    flag = movie_data.shape[0]

    movie_position = {}
    for i, row in enumerate(movie_data.values):
        movie_position.setdefault(str(int(row[0])), i)

    for row in rating_data.values:
        if user == row[0] - 1:
            flag -= 1
        else:
            if flag > 0:
                num_missing_sample += 1
            flag = movie_data.shape[0]
        user = int(row[0] - 1)
        movie_index = movie_position[str(int(row[1]))]
        rate = row[2]
        # print(f'user: [{user + 1}], movie: [{row[1]}], rate: {[rate]}')
        data[user][movie_index] = float(rate)


    print(f'missing sample rate: {num_missing_sample, num_missing_sample / data.shape[0]}')
    print(f'missing entry rate: {(data.shape[0] * data.shape[1] - rating_data.shape[0]) / (data.shape[0] * data.shape[1])}')

    new_data, missing_entry_rate = _filter(data, axis=1, missing_rate=0.9)

    print(f'new data: {new_data.shape}')
    print(f'missing entry rate: {missing_entry_rate}')
    missing_samples_num = 0
    for row in new_data:
        if len(row[row == 0]) > 0:
            missing_samples_num += 1
    print(f'missing samples rate : {missing_samples_num / new_data.shape[0]}')

    # exit(0)

    data_dir = osp.join(DATA_DIR, 'MovieLens1M')
    # save_path = osp.join(data_dir, 'new_data.npy')
    # np.save(save_path, new_data)
    # new_data = np.load(save_path)
    new_data = np.array(new_data, dtype=np.float32)
    new_data[new_data[:,:] == 0] = np.nan
    # print(f'Successfully the new data!')

    label = users_data.values[:, 2].reshape(-1, 1)
    data = np.concatenate((new_data, label), axis=1)
    print(f'data: {data.shape}')

    data[data[:, -1] == 56, -1] = 1
    data[data[:, -1] != 1, -1] = 0

    normal_data = data[data[:, -1] == 0]
    abnormal_data = data[data[:, -1] == 1]
    print(f'normal data: {normal_data.shape}')
    print(f'abnormal data: {abnormal_data.shape}')
    # print(f'a instance of normal data: {normal_data[0].tolist()}')
    # print(f'a instance of abormal data: {abnormal_data[0].tolist()}')
    save_path = osp.join(data_dir, 'complete_data_nan.npy')
    np.save(save_path, data)
    print(f'Sucessfullt save the [{save_path}]!')


def _filter(data, axis=1, missing_rate=0.8):

    new_data = []

    missing_entry_rate = 0
    if axis == 0:
        pass
    elif axis == 1:
        for i in range(data.shape[1]):
            column = data[:, i]
            column_missing_rate = len(column[column == 0]) / data.shape[0]
            if column_missing_rate <= missing_rate:
                new_data.append(data[:, i])
                missing_entry_rate += column_missing_rate
    else:
        raise Exception(f'Undefined axis [{axis}].')

    return np.transpose(np.array(new_data)), missing_entry_rate / len(new_data)



if __name__ == '__main__':
    


    pass