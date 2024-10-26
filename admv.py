__author__ = 'XF'
__date__ = '2023/09/06'

'''
Distribution transformation with missing values.
'''

from networks.main import build_networks
from optim.trainer import Trainer
from data.main import load_dataset

class ADMissingValue(object):
    
    def __init__(self, dataset_name, net_name, data_path, optimizer_name: str, lr: float, epochs: int, batch_size: int, device: str, results_dir: str, 
                        print=None, in_channels=None, _lambda=None, beta=None, latent_dimension=None, missing_rate=0.0, entropy_reg_coe=None, mu=0, std=1,
                        r_max=None, r_min=None, stop_threshold=None, alpha=None, split=None, mechanism='mcar'):

        self.net = build_networks(net_name, in_channels=in_channels, mid_dim=latent_dimension)

        begin_epoch = 1
        end_epoch = begin_epoch + epochs

     
        self.dataset =  load_dataset(dataset_name, data_path, missing_rate=missing_rate, 
                                     split=split, mechanism=mechanism)

        self.ae_trainer = Trainer(optimizer_name, lr=lr, begin_epoch=begin_epoch, dataset_name=dataset_name, end_epoch=end_epoch, batch_size=batch_size, device=device, print=print,
        results_dir=results_dir, _lambda=_lambda, latent_dimension=latent_dimension, beta=beta, entropy_reg_coe=entropy_reg_coe, mu=mu, std=std, r_max=r_max, missing_rate=missing_rate,
        r_min=r_min, stop_threshold=stop_threshold, alpha=alpha)

        
    def train(self):

                
        results = self.ae_trainer.train(self.dataset, self.net)
        return results
