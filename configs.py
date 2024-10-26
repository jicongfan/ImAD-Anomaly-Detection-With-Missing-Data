__author__ = 'XF'
__date__ = '2023/09/01'


'''
Basic configurations for this project.
'''

from os import path as osp

# dir
ROOT_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(ROOT_DIR, 'data')
RESULTS_DIR = osp.join(ROOT_DIR, 'results')
LOG_DIR = osp.join(ROOT_DIR, 'log')
ARGUMENTS_DIR = osp.join(ROOT_DIR, 'arguments')
TRAINING_LOG = osp.join(LOG_DIR, 'training_log')