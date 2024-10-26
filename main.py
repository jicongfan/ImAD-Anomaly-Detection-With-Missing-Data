__author__ = ''
__date__ = '2023/09/05'

'''
the entrance of this project.
'''

import click
import torch
import os
import numpy as np
import random

from admv import ADMissingValue

# self-defined library
from tools import Log, create_argparser, args_to_dict, json_dump, json_load, new_dir
from configs import TRAINING_LOG, RESULTS_DIR, ARGUMENTS_DIR, DATA_DIR
################################################################################
# Command line arguments
################################################################################
@click.command()
@click.option('--dataset_name', type=click.Choice(['kdd', 'arrhythmia', 'adult', 'speech', 'usoskin', 'segerstolpe', 'botnet', 'titanic', 'movielens1m', 'bladder', 'seq2_heart']))
@click.option('--load_arguments', type=click.Path(), default=None,
              help='arguments JSON-file path (default: None).')
@click.option('--repeat', type=int, default=1, help='the repeat time for calculating the average metric.')

@click.option('--seed', type=int, default=-1, 
                help='random seed for parameters initialization.')
@click.option('--stop_threshold', type=float, default=None, 
                help='Stop error for Sinkhorn algorithm.')
@click.option('--alpha', type=float, default=None,
                help='the coefficent of loss term: pseudo-abnormal data reconstruction')
@click.option('--beta', type=float, default=None, 
                help='The coefficient of loss term: data imputation')
@click.option('--_lambda', type=float, default=None, 
                help='The coefficient of loss term: normal data reconstruction')                             
@click.option('--rsd', type=str, default=None, 
                help='save dir of the experimental resutls.')
@click.option('--latent_dimension', type=int, default=None, 
                help='')  
@click.option('--missing_rate', type=float, default=0.0, 
                help='the missing rate of dataset')
@click.option('--entropy_reg_coe', type=float, default=None,
                help='coefficient of entropy regularization term of sinkhorn')
@click.option('--mu', type=float, default=0.0,
                help='the mean of normal distribution.')
@click.option('--std', type=float, default=None,
                help='the standard deviation of normal distribution')
@click.option('--r_max', type=float, default=None,
                help='upper bound of sampling distribution')
@click.option('--r_min', type=float, default=None,
                help='lower bound of sampling distribution')
@click.option('--split', type=int, default=1,
                help='the rate of splitting of testing set: normal / abnormal.')
@click.option('--mechanism', type=str, default='mcar',
                help='missing mechanism')
def main(dataset_name, load_arguments, repeat, stop_threshold, _lambda, rsd, seed, beta, 
            latent_dimension, missing_rate, entropy_reg_coe, mu, std, r_max, r_min, 
            alpha, split, mechanism):

    # training args
    if load_arguments is not None:
        default_args = json_load(os.path.join(ARGUMENTS_DIR, load_arguments))

    args = create_argparser(default_args).parse_args()

    # training logs
    training_log = os.path.join(TRAINING_LOG, args.log_dir)
    if not os.path.exists(training_log):
        os.makedirs(training_log)
    mylogger = Log(training_log, log_name=['log'])
    print = mylogger.print

    # update args
    args.dataset_name = dataset_name
    args.repeat = repeat
    args.seed = seed
    args.missing_rate = missing_rate
    args.mu = mu
    args.split = split
    args.mechanism = mechanism

    

    if stop_threshold is not None:
        args.stop_threshold = stop_threshold

    if _lambda is not None:
        args._lambda = _lambda
    
    if rsd is not None:
        args.rsd = rsd
    
    if beta is not None:
        args.beta = beta
    
    if std is not None:
        args.std = std
    
    if r_max is not None:
        args.r_max = r_max
    
    if r_min is not None:
        args.r_min = r_min
    
    if alpha is not None:
        args.alpha = alpha

    if latent_dimension is not None:
        args.latent_dimension = latent_dimension
    
    if entropy_reg_coe is not None:
        args.entropy_reg_coe = entropy_reg_coe

    # training information

    print('Dataset: %s' % args.dataset_name)
    print('Network: %s' % args.net_name)
    print(f'Latent dimension: {args.latent_dimension}')
    print(f'Split: [Normal:Abnormal={args.split}:1]')
    print('About Optimization ===============================')
    print(f'Pptimizer:{args.optimizer_name}')
    print(f'Learning rate:[{args.lr}]')
    print(f'Epochs:[{args.epochs}]')
    print(f'Batch_size:[{args.batch_size}]')
    
    print(f'Hyper-parameters ================================')
    print(f'Alpha: [{args.alpha}]')
    print(f'Beta: [{args.beta}]')
    print(f'Lambda: [{args._lambda}]')

    print(f'Sinkhorn Entropy Coefficient: [{args.entropy_reg_coe}]')
    print(f'Stop threshold: [{args.stop_threshold}]')

    print(f'Experiment setting ===============================')
    print(f'Negative samples from: N({args.mu, args.std})')
    print(f'Missing Mechanism: {args.mechanism}')
    print(f'Missing rate: [{missing_rate * 100}%]')
    print(f'Min Radius: [{args.r_min}]')
    print(f'Max Radius: [{args.r_max}]')
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    print(f'Computation device: {device}')
    print(f'Repeat time: [{args.repeat}]')

    in_channels = None

    if dataset_name == 'kdd':
        in_channels = 121
    elif dataset_name == 'adult':
        in_channels = 14
    elif dataset_name == 'arrhythmia':
        in_channels = 274
    elif dataset_name == 'speech':
        in_channels = 400
    elif dataset_name == 'usoskin':
        in_channels = 25334
    elif dataset_name == 'segerstolpe':
        in_channels = 1000
    elif dataset_name == 'botnet':
        in_channels = 115
    elif dataset_name == 'titanic':
        in_channels = 9
    elif dataset_name == 'movielens1m':
        in_channels = 498
    elif dataset_name == 'bladder':
        in_channels = 23341
    elif dataset_name == 'seq2_heart':
        in_channels = 23341
    else:
        raise Exception(f'Unknown dataset name [{dataset_name}]!')
    
    
    if args.rsd is not None:
        results_dir = new_dir(os.path.join(RESULTS_DIR, args.rsd))
    else:
        results_dir = new_dir(RESULTS_DIR)
    
    # save arguments
    arguments_path = os.path.join(results_dir, '%s.json' % dataset_name)
    json_dump(arguments_path, args_to_dict(args, list(default_args.keys())))

    for i in range(1, args.repeat + 1):
        if args.seed != -1:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            print('Set seed to [%d].' % args.seed)
        print(f'================== the {i}-th time ==================')
        model = ADMissingValue(
                dataset_name=args.dataset_name,
                net_name=args.net_name,
                data_path=DATA_DIR,
                lr=args.lr,
                optimizer_name=args.optimizer_name,
                results_dir=args.rsd,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                print=print,
                in_channels=in_channels,
                _lambda=args._lambda,
                beta=args.beta,
                latent_dimension=args.latent_dimension,
                missing_rate=args.missing_rate,
                entropy_reg_coe=args.entropy_reg_coe,
                mu=args.mu,
                std=args.std,
                r_min=args.r_min,
                r_max=args.r_max,
                stop_threshold=args.stop_threshold,
                alpha=args.alpha,
                split=args.split,
                mechanism=mechanism,
                )
        results = model.train()


    mylogger.ending


if __name__ == '__main__':

    main()
    