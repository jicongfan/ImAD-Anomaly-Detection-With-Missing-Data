
from .mlp import MVNet


def build_networks(net_name, in_channels=3, mid_dim=128):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mlp')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mlp':
        net = MVNet(input_dim=in_channels, mid_dim=mid_dim)
    
    return net
