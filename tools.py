__author__ = 'XF'
__date__ = '2023/07/03'

'''
some frequently used general functions for FairAD.

'''
import os
from os import path as osp
import pickle
import time
import numpy as np
from builtins import print as b_print
import json 
from argparse import ArgumentParser, ArgumentTypeError
from scipy import optimize
import torch

ROOT_DIR = osp.abspath(osp.dirname(__file__))
DATA_DIR = osp.join(ROOT_DIR, 'data')

# other
begining_line = '=============================== Begin ======================================='
ending_line =   '================================ End ========================================'


# object serialization
def obj_save(path, obj):

    if obj is not None:
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    else:
        print('object is None!')


# object instantiation
def obj_load(path):

    if os.path.exists(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    else:
        raise OSError('no such path:%s' % path)

# logging
class Log(object):

    def __init__(self, log_dir, log_name):
        self.log_path = osp.join(log_dir, generate_filename('.txt', *log_name, timestamp=True))
        self.print(begining_line)
        self.print('date: %s' % time.strftime('%Y/%m/%d-%H:%M:%S'))
    
    def print(self, *args, end='\n'):

        with open(file=self.log_path, mode='a', encoding='utf-8') as console:
            b_print(*args, file=console, end=end)
        b_print(*args, end=end)
    
    @property
    def ending(self):
        self.print('date: %s' % time.strftime('%Y/%m/%d-%H:%M:%S'))
        self.print(ending_line)



def generate_filename(suffix, *args, sep='_', timestamp=False):

    '''

    :param suffix: suffix of file
    :param sep: separator, default '_'
    :param timestamp: add timestamp for uniqueness
    :param args:
    :return:
    '''

    filename = sep.join(args).replace(' ', '_')
    if timestamp:
        filename += time.strftime('_%Y%m%d%H%M%S')
    if suffix[0] == '.':
        filename += suffix
    else:
        filename += ('.' + suffix)

    return filename


def json_load(path):
    
    with open(path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def json_dump(path, dict_obj):

    with open(path, 'a+', encoding='utf-8') as f:
        json.dump(dict_obj, f, indent=4, ensure_ascii=False)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def create_argparser(default_args):

    parser = ArgumentParser()
    add_dict_to_argparser(parser, default_args)
    return parser


def new_dir(father_dir, mk_dir=None):

    if mk_dir is None:
        new_path = osp.join(father_dir, time.strftime('%Y%m%d%H%M%S'))
    else:
        new_path = osp.join(father_dir, mk_dir)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path
    

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("boolean value expected")



# def get_missing_data(data, missing_rate):

#     mask = np.random.rand(*data.shape) < missing_rate

#     data[mask] = 0.0

#     return data, 1 - 1 * mask


def get_missing_data(data, missing_rate, mechanism='mcar'):


    if mechanism == 'mcar':
        mask = np.random.rand(*data.shape) < missing_rate # True for missing values, false for others
        mask = torch.from_numpy(mask)
    elif mechanism == 'mar':
        mask = MAR_mask(data, p=missing_rate)

    elif mechanism == 'mnar':
        mask = MNAR_mask_quantiles(data, p=missing_rate)

    data[mask] = 0.0

    return data, 1 - 1 * mask


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

