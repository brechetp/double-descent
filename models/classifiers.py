import torch.nn as nn
import torch
import utils
import numpy as np

class FCN(nn.Module):
    '''Fully-connected neural network'''

    def __init__(self, input_dim, num_classes, width=[1024], lrelu=0.01):

        super().__init__()

        if type(width) is int:
            width = [width]
        sizes = [input_dim, *width, 10]
        #mlp = utils.construct_mlp_net(sizes, fct_act=nn.LeakyReLU, kwargs_act={'negative_slope': lrelu, 'inplace': True})
        mlp = utils.construct_mlp_net(sizes, fct_act=nn.ReLU, args_act=[True])

        #self.main = nn.Sequential(mlp, nn.Softmax())
        self.main = mlp

        return




    def forward(self, x):

        #vec = torch.cat((is_man.view(-1, 1), x.view(x.size(0), -1)), dim=1)

        out = self.main(x.view(x.size(0), -1))
        return out


    def num_parameters(self, only_require_grad=False):
        '''Return the number of parameters in the model'''
        return sum(p.numel() for p in self.parameters() if not only_require_grad or p.requires_grad)


class FCN3(FCN):

    def __init__(self, input_dim, num_classes, min_width, max_width, interpolation='linear'):

        super().__init__()

        if interpolation == 'linear':
            # linear interpolation between the two extrema
            widths = np.linspace(min_widht, max_width+1, 3)  # need t
