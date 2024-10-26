
from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from optim.sinkhorn import SinkhornDistance
from os import path as osp
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter as sw
import itertools
# self-defined library


class Trainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, begin_epoch: int = 1, end_epoch: int = 100,
                 batch_size: int = 128, device: str = 'cuda', print=None, dataset_name=None, results_dir=None, 
                 _lambda=None, latent_dimension=None, stop_threshold=None, entropy_reg_coe=None, beta=None, mu=0, 
                 std=1, r_min=None, r_max=None, missing_rate=0.0, alpha=None):

        super().__init__(optimizer_name, lr, end_epoch - begin_epoch, None, batch_size, device)


        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch
        self._lambda = _lambda
        self.beta = beta
        self.mu = mu
        self.std = std
        self.r_max = r_max
        self.r_min = r_min
        self.missing_rate = missing_rate
        self.alpha = alpha
 
        self.print = print
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.results_dir = results_dir
        self.latent_dimension = latent_dimension
        self.target_distribution_sampling_epoch = 1
        self.test_epoch = 5
        self.results = {
            'auroc': 0.0,
            'auprc': 0.0,
            "Time": None 
        }

        self.sinkhorn_loss = SinkhornDistance(eps=entropy_reg_coe, max_iter=int(1e3), thresh=stop_threshold, device=self.device)


    def train(self, dataset: BaseADDataset, net: BaseNet):
        
        # tensorboardX saving loss
        loss_writer = sw()

        # Set device for network
        net = net.to(self.device)
        # Set optimizer
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.lr)
        else:
            raise Exception(f'Unknown optimizer name [{self.optimizer_name}].')
        
        
        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size)

        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, max_lr=1e-4, step_size_up=len(train_loader), mode='triangular')

        # Training
        print('Starting training...')
        net.train()
        step = 1
        start = time.time()
        with trange(self.begin_epoch, self.end_epoch) as pbar:
            for epoch in pbar:
                loss_epoch = 0.0
                n_batches = 0
                # normal samples ===========
                nor_imputed_loss_batch = 0.0
                nor_recon_loss_batch = 0.0
                nor_dist_loss_batch = 0.0
                # generated negative samples =========
                neg_imputed_loss_batch = 0.0
                neg_dist_loss_batch = 0.0

                for data in train_loader:
                    inputs, _, _, masks = data
                    inputs = inputs.to(self.device)
                    masks = masks.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    imputed_data, mid_repre, recover_data = net(inputs)

                    # normal data ================================================================================================
                    # imputation loss
                    imputed_loss = torch.sum(torch.mul((imputed_data  - inputs), masks) ** 2, dim=tuple(range(1, inputs.dim()))) * self.beta
                    loss = torch.mean(imputed_loss)
                    nor_imputed_loss_batch += loss.item()
                    # loss_writer.add_scalar('%s_%s_normal_imputation_loss' % (self.dataset_name, str(self.latent_dimension)), loss.item(), step)

                    # projection loss
                    targets = target_distribution_sampling(self.batch_size, mid_repre[0].shape, r_min=0.0, r_max=self.r_min, mu=0.0, std=0.5)
                    targets = targets.to(self.device)
                    dist_loss = self.sinkhorn_loss(mid_repre, targets)
                    nor_dist_loss_batch += dist_loss.item()
                    loss += dist_loss
                    # loss_writer.add_scalar('%s_%s_normal_dist_loss' % (self.dataset_name, str(self.latent_dimension)), dist_loss.item(), step)
                    
                    # reconstruction loss term
                    recon_loss = torch.mean(torch.sum(torch.mul((recover_data - inputs), masks) ** 2, dim=tuple(range(1, inputs.dim())))) * self._lambda
                    nor_recon_loss_batch += recon_loss.item()
                    # loss_writer.add_scalar('%s_%s_normal_recon_loss' % (self.dataset_name, str(self.latent_dimension)), recon_loss.item(), step)
                    loss += recon_loss

                    loss.backward()
                    optimizer.step()

                    # loss_writer.add_scalar('%s_%s_loss' % (self.dataset_name, str(self.latent_dimension)), loss.item(), step)
                    step += 1
                    loss_epoch += loss.item()
                    n_batches += 1
                    # scheduler.step()
                
                pbar.set_description(
                    f'Loss: {loss_epoch / n_batches:.4f}')
                    
                if epoch % self.test_epoch == 0:
                    self.print(f'\nepoch:[{epoch}]#############################')
                    self.test(dataset, net)
                    net.train()
                    
        self.results['Time'] = time.time() - start
        print(f'using time: {self.results["Time"]}')
        print('Finished training.')

        return self.results


    def test(self, dataset: BaseADDataset, net: BaseNet):

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size)
        self.print('model testing ...')

        score = None
        idx_label_score = []
        net.eval()
        all_mid_repre = None
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx, _ = data
                inputs = inputs.to(self.device)

                _, mid_repre, _ = net(inputs)
                # save the test point in the latent space
                if all_mid_repre is None:
                    all_mid_repre = mid_repre
                else:
                    all_mid_repre = torch.concat((all_mid_repre, mid_repre))

                score = torch.sqrt(torch.sum(mid_repre ** 2, dim=tuple(range(1, mid_repre.dim()))))
                
                idx_label_score += list(zip(
                    idx.cpu().data.numpy().tolist(),
                    labels.cpu().data.numpy().tolist(),
                    score.cpu().data.numpy().tolist()
                    ))


        _, labels, scores = zip(*idx_label_score)
        
        # detection performance =========================================
        # AUROC
        auroc = roc_auc_score(labels, scores)
        self.print('Test set AUROC: [{:.2f}%]'.format(100. * auroc))
        
        # AUPRC
        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = auc(recall, precision)
        self.print('Test set AUPRC: [{:.2f}%]'.format(100. * auprc))




def target_distribution_sampling(size, sample_dim, r_max=None, r_min=0.0, mu=0, std=1):

    '''
    :params
    size:
    sample_dim: the dimension of the sample from the restricted distribution
    mu: the mean of normal distribution
    std: the standard devariate of normal distribution

    '''

    Sampler = randn
    targets = None
    
    while size > 0:

        sample = Sampler(sample_dim, mean=mu, std=std)
        sample_norm = torch.sqrt(torch.sum(sample ** 2))

        if r_min < sample_norm < r_max:
            if targets is None:
                targets = sample.unsqueeze(0)
            else:
                targets = torch.cat((targets, sample.unsqueeze(0)))
            size -= 1
    return targets


def randn(sample_dim, mean=0.0, std=1.0):

    '''
    N(0, 1) Gaussian
    '''
    return torch.distributions.Normal(loc=mean, scale=std).sample(sample_dim).squeeze(-1)

