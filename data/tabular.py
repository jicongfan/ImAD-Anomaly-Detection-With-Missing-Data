
'''
the script is for tabular datasets.
'''
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from base.torchvision_dataset import TorchvisionDataset
from scipy import optimize

class CustomDataset(Dataset):

    def __init__(self, data, labels, mask):
        self.data = data
        self.labels = labels
        self.mask = mask
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _data = torch.from_numpy(self.data[idx]).type(torch.FloatTensor)
        m = torch.from_numpy(self.mask[idx]).type(torch.FloatTensor)
        return _data, self.labels[idx], idx, m


class TabularDataset(TorchvisionDataset):

      def __init__(self, root, dataset_name=None, missing_rate=0.0, split=None, mechanism='mcar'):
        super().__init__(root)
        self.name='tabular'
        self.n_classes = 2  # 0: normal, 1: outlier
        self.missing_rate = missing_rate # for real incomplete data

        path = os.path.join(root, '%s/complete_data.npy' % dataset_name)
        if dataset_name in ['Titanic', 'MovieLens1M', 'Bladder', 'Seq2_Heart']:
            missing_data = np.array(np.load(os.path.join(root, '%s/complete_data_nan.npy' % dataset_name), allow_pickle=True), dtype=np.float64)
            mask = 1 - 1 * np.isnan(missing_data[:,:-1])
            self.missing_rate = np.count_nonzero(1 - mask) / ((mask.shape[0] * (mask.shape[1])) * 1.0)
            print(f'missing rate: {self.missing_rate}')
            missing_data, mask = get_missing_data(path, missing_rate, mechanism=mechanism)
        else:
            missing_data, mask = get_missing_data(path, missing_rate, mechanism=mechanism)

        train_data, train_lab, train_mask, test_data, test_lab, test_mask = split_train_test_data(missing_data, mask, split=split)
        

        train_data = np.array(train_data, dtype=np.float64)
        test_data = np.array(test_data, dtype=np.float64)

        print('======== dataset info =========')
        print(f'train data: {train_data.shape}')
        print(f'test data: {test_data.shape}')
        print('======== dataset info =========')
        self.train_set = CustomDataset(train_data, train_lab, train_mask)

        self.test_set = CustomDataset(test_data, test_lab, test_mask)



def get_missing_data(path, missing_rate, mechanism='mcar'):


    complete_data = np.random.permutation(np.load(path, allow_pickle=True)) * 1.0

    if mechanism == 'mcar':
        mask = np.random.rand(*complete_data[:, :-1].shape) < missing_rate # True for missing values, false for others
        
    elif mechanism == 'mar':
        mask = MAR_mask(complete_data[:, :-1], p=missing_rate)

    elif mechanism == 'mnar':
        mask = MNAR_mask_quantiles(complete_data[:, :-1], p=missing_rate)

    miss_data = np.copy(complete_data)
    miss_data[:, :-1][mask] = np.nan

    return miss_data, 1 - 1*mask


def split_train_test_data(data, mask, split=1):

    data = np.array(data, dtype=np.float64)
    normal_data = data[data[:, -1] == 0]
    normal_mask = mask[data[:, -1] == 0]
    abnormal_data = data[data[:, -1] == 1]
    abnormal_mask = mask[data[:, -1] == 1]
    assert len(normal_data) > len(abnormal_data)

    num_train_data = len(normal_data) - len(abnormal_data)
    train_data = normal_data[:num_train_data, :-1]
    train_mask = normal_mask[:num_train_data]
    test_normal_data = normal_data[num_train_data:]
    test_normal_mask = normal_mask[num_train_data:]

    train_lab = np.zeros(len(train_data))

    num_test_abnormal_data = int(len(test_normal_data) / split)
    test_abnormal_data = abnormal_data[:num_test_abnormal_data, :-1]
    test_abnormal_mask = abnormal_mask[:num_test_abnormal_data]
    test_data = np.concatenate((test_normal_data[:, :-1], test_abnormal_data))
    test_mask = np.concatenate((test_normal_mask, test_abnormal_mask))
    test_lab = np.concatenate((np.zeros(len(test_normal_data)), np.ones(len(test_abnormal_data))))
    
    # print('======== test set info =========')
    # print(f'normal data: {test_normal_data[:,:-1].shape}')
    # print(f'abnormal data: {test_abnormal_data.shape}')
    # print('======== test set info =========')

    # normalization
    mu = np.nanmean(train_data, axis=0)
    std = np.nanstd(train_data, axis=0)
    std[std == 0] = 1

    train_data = (train_data - mu) / std
    test_data = (test_data - mu) / std

    train_data[np.isnan(train_data)] = 0.0
    test_data[np.isnan(test_data)] = 0.0

    return train_data, train_lab, train_mask, test_data, test_lab, test_mask



def MAR_mask(X, p, p_obs=0.3):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def MNAR_mask_quantiles(X, p, q=0.75, p_params=0.7, cut='both', MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    q : float
        Quantile level at which the cuts should occur

    p_params : float
        Proportion of variables that will have missing values

    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        
    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1) ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = quantile(X[:, idxs_na], 1-q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = quantile(X[:, idxs_na], 1-q, dim=0)
        l_quants = quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts

