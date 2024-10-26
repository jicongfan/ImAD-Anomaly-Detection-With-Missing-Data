
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import get_missing_data

class Imputer(nn.Module):

    def __init__(self, input_dim):
        super(Imputer, self).__init__()

        self.input_dim = input_dim

        # self.impute = nn.Sequential(

        #     nn.Linear(self.input_dim, 512, bias=False),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(512, 128, bias=False),
        #     nn.LeakyReLU(inplace=True),

        #     nn.Linear(128, 128, bias=False),
        #     nn.LeakyReLU(inplace=True),

        #     nn.Linear(128, 512, bias=False),
        #     nn.LeakyReLU(inplace=True),

        #     nn.Linear(512, self.input_dim, bias=False),
        # )
        self.impute = nn.Sequential(
                nn.Linear(self.input_dim, 512, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(512, 512, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(512, self.input_dim, bias=False),
            )


    def forward(self, x):

        return self.impute(x)


class Projector(nn.Module):

    def __init__(self, input_dim, mid_dim):
        super(Projector, self).__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        
        # self.project = nn.Sequential(
        #     nn.Linear(self.input_dim, 512, bias=False),
        #     nn.LeakyReLU(inplace=True),

        #     nn.Linear(512, 256, bias=False),
        #     nn.LeakyReLU(inplace=True),

        #     nn.Linear(256, 128, bias=False),
        #     nn.LeakyReLU(inplace=True),

        #     nn.Linear(128, self.mid_dim, bias=False),

        # )
        self.project = nn.Sequential(
                nn.Linear(self.input_dim, 512, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(512, 256, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(256, self.mid_dim, bias=False),
            )

    def forward(self, x):

        return self.project(x)


class Recover(nn.Module):

    def __init__(self, mid_dim, recover_dim) -> None:
        super(Recover, self).__init__()
        self.mid_dim = mid_dim
        self.recover_dim = recover_dim


        # self.decoder = nn.Sequential(

        #     nn.Linear(self.mid_dim, 128, bias=False),
        #     nn.LeakyReLU(inplace=True),

        #     nn.Linear(128, 512, bias=False),
        #     nn.LeakyReLU(inplace=True),

        #     nn.Linear(512, self.recover_dim, bias=False),

        # )

        self.decoder = nn.Sequential(
                nn.Linear(self.mid_dim, 200, bias=False),
                nn.LeakyReLU(inplace=True),

                nn.Linear(200, self.recover_dim, bias=False)
            )

    def forward(self, x):
        
        return self.decoder(x)


class MVNet(nn.Module):

    def __init__(self, input_dim, mid_dim):
        super(MVNet, self).__init__()

        self.imputer = Imputer(input_dim=input_dim)
        self.projector = Projector(input_dim=input_dim, mid_dim=mid_dim)
        self.recover = Recover(mid_dim=mid_dim, recover_dim=input_dim)

    def forward(self, x, negative=False, missing_rate=0.0, mechanism=None):

        if not negative:
            imputed_data = self.imputer(x)
            mid_repre = self.projector(imputed_data)
            recover_data = self.recover(mid_repre)
            return imputed_data, mid_repre, recover_data
        else:
            recover_data = self.recover(x)
            
            missing_data, masks = get_missing_data(recover_data.cpu().data, missing_rate, mechanism)
            missing_data = missing_data.to('cuda')
            imputed_data = self.imputer(missing_data)
            mid_repre = self.projector(imputed_data)
            return imputed_data, mid_repre, missing_data, masks
        
        
