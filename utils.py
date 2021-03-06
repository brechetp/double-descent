import torchvision
import torch
from datasets.rsna import RSNADataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision.datasets import MNIST, CIFAR10
import torch.nn as nn
from collections import OrderedDict
from subprocess import Popen, PIPE
import re
import PIL
import math



def pad_collate_fn(batch):
    '''collate function for images of different sizes'''
    data, man, age = zip(*batch)
    sizes = torch.tensor([item[0].size()[1:] for item in batch])
    bounds = torch.max(sizes, dim=0)[0]
    margins = bounds.view(1, -1) - sizes
    pad_sizes = [(margin_w//2, margin_w - margin_w//2, margin_h//2, margin_h - margin_h//2) for (margin_h, margin_w) in margins]
    data = torch.cat([nn.functional.pad(item[0].unsqueeze(0), pad_s) for item, pad_s in zip(batch, pad_sizes)])

    return (data, torch.tensor(man, dtype=int), torch.tensor(age, dtype=torch.float))





def get_dataset(dataset='rsna', dataroot='data/', imresize=None, augment=None, normalize=False, transform=None):


    num_chs = 1 if dataset.lower() in [ 'rsna', 'mnist' ] else 3

    if transform is None:
        transform_lst = []


        if imresize is not None:
            transform_lst.append(transforms.Resize(imresize))

        transform_lst.append(transforms.ToTensor())

        if normalize:
            transform_lst.append(transforms.Normalize(num_chs*(0.5,), num_chs*(0.5,)))


        transform = transforms.Compose(transform_lst)

    if dataset.lower() == 'rsna':


        train_dataset  = RSNADataset(dataroot, transform)
        test_dataset = RSNADataset(dataroot, transform, train=False)


    elif dataset.lower() == 'mnist':

        train_dataset = MNIST(dataroot, train=True, transform=transform, download=True)
        test_dataset = MNIST(dataroot, train=False, transform=transform, download=True)

    elif dataset.lower() == 'cifar10':

        train_dataset = CIFAR10(dataroot, train=True, transform=transform, download=True)
        test_dataset = CIFAR10(dataroot, train=False, transform=transform, download=True)



    return train_dataset, test_dataset, num_chs



def get_dataloader(train_dataset, test_dataset, batch_size, ss_factor=1., size_max=None, collate_fn=None):
    '''ss_factor: for the valid set'''

    train_idx = torch.randperm(len(train_dataset))
    train_size = int(len(train_dataset) * ss_factor)

    val_size = len(train_dataset) - train_size

    if size_max is not None:
        train_size = min(size_max, train_size)

    if test_dataset is None:
        test_size = None
    else:
        test_idx = torch.randperm(len(test_dataset))
        test_size = len(test_dataset)

    #  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                              sampler=SubsetRandomSampler(train_idx[:train_size]),
                              collate_fn=collate_fn)  # the first

    val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                            sampler=SequentialSampler(train_idx[train_size:]),
                            collate_fn=collate_fn ) if val_size >0 else None # the rest of the indices

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn ) if not(test_dataset is None) else None

    return train_loader,train_size, val_loader, val_size, test_loader, test_size


def construct_mlp_layers(sizes, fct_act=nn.ReLU, args_act=[], kwargs_act={}, args_linear=[], kwargs_linear={}, tanh=False, out_layer=True, batchnorm=False, batchnorm_in=False):
    '''Constructs layers  with each layer being of sizes[i]
    with specified init and end size in sizes[0] sizes[-1]

    Args:
        sizes (list): the list [nin, *layers, nout] of dimensions for the layers
        layer (nn.): the linear transformation for the layers (default: nn.Linear)
        fct_act: the non linear activation function (default: nn.ReLU)
        tanh (bool): if true will append a tanh activation layer as the last layer (to map output to -1, 1)
        batchnorm (bool): if true, use a batch norm

    return: ordered dict of layers
'''

    idx = 0
    size = sizes[0]
    layers = []
    norm_layer = nn.BatchNorm1d

    for idx, new_size in enumerate(sizes[1:-1] if out_layer else sizes[1:], 1):  # for all layers specified in sizes
        # switch the sizes
        prev_size, size = size, new_size
        # adds the layer
        layers.append((
                '{}{}-{}-{}'.format(nn.Linear.__name__, idx, prev_size, size),
                nn.Linear(prev_size, size, *args_linear, **kwargs_linear)
                ))
        if batchnorm:
            if idx == 1 and not out_layer and not batchnorm_in:  # for discriminator network, no normalization at the beginning
                pass
            #  elif out_layer and not tanh and idx == len(sizes) - 2:  # for generator, no batchnorm at the last layer (following DCGAN)
                #  pass
            else:
                layers.append((
                    '{}{}-{}'.format(norm_layer.__name__, idx, size),
                    norm_layer(size),
                    ))

        # adds the non linear activation function
        layers.append((
                '{}{}-{}'.format(fct_act.__name__, idx, size),
                fct_act(*args_act, **kwargs_act)
                ))

    # at  the end of the loop, we still have the last layer to add, but without
    # activation
    if out_layer:
        layers.append((
            '{}{}-{}-{}'.format(nn.Linear.__name__, idx+1, size, sizes[-1]),
            nn.Linear(size, sizes[-1], *args_linear, **kwargs_linear)
            ))

        if tanh:
            layers.append((
                'tanh', nn.Tanh()
                ))

    return OrderedDict(layers)

def construct_mlp_net(sizes, fct_act=nn.ReLU, args_act=[], kwargs_act={}, args_linear=[], kwargs_linear={}, tanh=False, out_layer=True, batchnorm=False, batchnorm_in=False):
    layers = construct_mlp_layers(sizes, fct_act, args_act, kwargs_act, args_linear, kwargs_linear, tanh, out_layer, batchnorm, batchnorm_in)
    return nn.Sequential(layers)

def num_parameters(model, only_require_grad=False):
        '''Return the number of parameters in the model'''
        return sum(p.numel() for p in model.parameters() if not only_require_grad or p.requires_grad)

def get_norm_weights(model):

    norm_squared = 0.
    N = 0  # the total number of parameters
    for p in model.parameters():
        norm_squared += p.pow(2).sum()
        N += p.numel()

    return (norm_squared/N).sqrt()

def parse_transform(fname, *args):
    '''Returns the transform if any'''

    process = Popen(['grep', 'Resize', fname], stdout=PIPE, stderr=PIPE)
    lines = process.communicate()[0].decode('utf-8').strip().splitlines()
    size = None
    transform_lst = []
    depth = 0
    TRANSFORMS = {
        'RandomAffine' : transforms.RandomAffine,
        'Resize': transforms.Resize,
        'ToTensor': transforms.ToTensor,
        'Normalize': transforms.Normalize,
    }
    with open(fname, 'r') as _f:
        for line in _f:
            if depth == 0:
                if line.find('Transform: Compose') != -1:
                    depth += 1
            elif depth == 1:
                depth += line.count("(")
                depth -= line.count(")")
                if depth == 0:
                    break
                par_pos = line.find('(')
                tfm_name = line.split('(')[0].strip()
                tfm_args = line[par_pos+1:].strip().rstrip(')') # leave trailing )
                args, kwargs = parse_layer_args(tfm_args)
                transform_lst.append(TRANSFORMS[tfm_name](*args, **kwargs))

    transform = transforms.Compose(transform_lst)

    return transform

def get_image_resize(transform):
    '''Return the image from the tranform if any'''

    tfms = transform.transforms
    resize =None
    for t in tfms:
        if type(t) is transforms.Resize:
            resize = t.size  # warning, what if tuple
    return resize

def get_ouput_dim_conv(d_in, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1)):
    '''Assume tuple input'''
    d_out =  tuple( int(math.floor((d + 2 * p - dil*(k -1) - 1) / s + 1)) for d,p,dil,k,s in zip(d_in, padding, dilation, kernel_size, stride))
    return d_out

def get_ouput_dim_conv(d_in, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), output_padding=(0,0)):
    d_out =  tuple( ((d-1) * s  - 2*p + dil * (k - 1 ) + out_p+1) for d,s,p,dil,k,out_p in zip(d_in, stride, padding, dilation, kernel_size, output_padding))
    return d_out

def parse_archi(fname, *args):
    '''Parse a log file containing architectures for networks'''


    nets = {}

    net_re = re.compile('Model:')
    LAYERS = {
        'Linear': nn.Linear,
        'BatchNorm1d': nn.BatchNorm1d,
        'LeakyReLU': nn.LeakyReLU,
        'ConvTranspose2d': nn.ConvTranspose2d,
        'Conv2d': nn.Conv2d,
        'BatchNorm2d': nn.BatchNorm2d,
        'Tanh': nn.Tanh,
        'Sigmoid': nn.Sigmoid,
        'Bilinear': nn.Bilinear,
        'ReLU': nn.ReLU,
        'ELU': nn.ELU,
    }

    depth = 0
    with open(fname, 'r') as _f:
        for line in _f:
            if depth == 0:
                new_net = net_re.match(line)
                if new_net:
                    c_depth = 1
                    net = dict()
                    net_name = line.split(' ')[1].rstrip('(')
                    depth += line.count("(")
                    depth -= line.count(")")

            elif depth == 1:  # definition of the modules (attributes of the network)

                fields = line.split(':')
                if len(fields) == 2:
                    # should always be the case
                    # not the case when closing the sequential module
                    par_pos = fields[1].find('(')
                    module_name = fields[0].strip().rstrip(')').lstrip('(')
                    # will me main, etc.
                    module_type = fields[1][:par_pos].strip()  # Sequential

                    if module_type == 'Sequential':
                        nn_module_type = nn.Sequential
                        nn_layers = []
                    else:
                        raise NotImplementedError

                depth += line.count("(")
                depth -= line.count(")")


            elif depth == 2:

                depth += line.count("(")
                depth -= line.count(")")


                fields = line.split(':')

                if len(fields) == 2:
                    layer_name = fields[0].strip().lstrip('(').rstrip(')')

                    layer_type = fields[1][:fields[1].find('(')].strip()
                    layer_args = fields[1][fields[1].find('(')+1:].strip().rstrip(')') # leave trailing )
                    args, kwargs = parse_layer_args(layer_args)
                    try:
                        nn_layers.append((layer_name, LAYERS[layer_type](*args, **kwargs)))
                    except TypeError as e:
                        print(e)
                        print(layer_type)
                if depth == 1:  # end of sequential
                    net[module_name] = nn_module_type(OrderedDict(nn_layers)) # is sequential
                    nn_layers = []


    return net


def construct_FCN(archi):

    class FCN(nn.Module):

        def __init__(self, main):
            super().__init__()
            self.main = main

        def forward(self, x):

            return self.main(x.view(x.size(0), -1))

    return FCN(archi['main'])



def construct_G(archi_G, image_size):
    '''construct the nn.Modules from the dict of archis'''

    class Generator(nn.Module):

        def __init__(self, fc=None, deconv=None, deconv_dim=None, depth=None, image_size=None, inter_shape=None, decoder=None):
            super().__init__()
            if decoder is not None:
                self.decoder = Decoder(decoder)
                self.forward = self._forward_decoder
            else:
                self.fc = fc
                self.deconv = deconv
                self.forward = self._forward_gen
                self.inter_shape = (deconv_dim, image_size // depth, image_size // depth) if inter_shape is None else inter_shape

        def _forward_gen(self, z):
            out = self.fc(z)
            out = out.view(z.shape[0], *self.inter_shape)
            out = self.deconv(out)
            return out

        def _forward_decoder(self, z):
            return self.decoder(z)

    class Decoder(nn.Module):
        '''Simple decoder module'''

        def __init__(self, main):
            super().__init__()
            self.main = main

        def forward(self, z):
            out = self.main(z.view(z.size(0), z.size(1), 1, 1))
            return out

    class GeneratorGP(nn.Module):

        def __init__(self, block1,  block2, deconv_out, preprocess, act, DIM=64, IM_DIM=(1, 28, 28)):
            super().__init__()
            self.block1 = block1
            self.block2 = block2
            self.deconv_out = deconv_out
            self.preprocess = preprocess
            self.sigmoid = nn.Sigmoid()
            self.DIM = DIM
            self.IM_DIM = IM_DIM

        def forward(self, z):

            output = self.preprocess(z)
            output = output.view(-1, 4*self.DIM, 4, 4)
            #print output.size()
            output = self.block1(output)
            #print output.size()
            output = output[:, :, :7, :7]
            #print output.size()
            output = self.block2(output)
            #print output.size()
            output = self.deconv_out(output)
            output = self.sigmoid(output)
            #print output.size()
            return output.view(-1, *self.IM_DIM)



    if 'block1' in archi_G.keys():
        return GeneratorGP(archi_G['block1'], archi_G['block2'], archi_G['deconv_out'], archi_G['preprocess'], archi_G['sigmoid'])
    elif archi_G['module_type'] == 'Sequential':
        return Generator(archi_G['fc'], archi_G['deconv'], archi_G['dim'], archi_G['depth'], image_size, archi_G.get('inter_shape', None))
    elif archi_G['module_type']== 'Decoder':
        return Generator(decoder=archi_G['main'])

def construct_F(archi_F, image_size):
    '''construct the nn.Modules from the dict of archis'''

    class Discriminator(nn.Module):

        def __init__(self, conv=None, fc=None,  conv_dim=None, depth=None, image_size=None, inter_dim=None, encoder=None):
            super().__init__()
            if encoder is not None:
                self.encoder = Encoder(encoder)
                self.forward = self._forward_encoder
            else:
                self.conv = conv
                self.fc = fc
                self.depth = depth
                self.forward = self._forward
                self.inter_dim = conv_dim * (image_size // depth) * (image_size // depth) if inter_dim is None else inter_dim

        def _forward(self, x):
            out = self.conv(x)
            out = out.view(x.shape[0], self.inter_dim)
            out = self.fc(out)
            return out

        def _forward_encoder(self, x):
            return self.encoder(x)

    class Encoder(nn.Module):
        '''Simple decoder module'''

        def __init__(self, main):
            super().__init__()
            self.main = main

        def forward(self, x):
            out = self.main(x).view(x.size(0), -1)
            return out

    class DiscriminatorGP(nn.Module):

        def __init__(self, main, output, DIM=64, IM_DIM=(1,28,28)):

            super().__init__()
            self.main = main
            self.output = output
            self.DIM = DIM
            self.IM_DIM = IM_DIM

        def forward(self, x):

            x = x.view(-1, *self.IM_DIM)
            out = self.main(x)
            out = out.view(-1, 4*4*4*self.DIM)
            out = self.output(out)  # might be id

            return out

    if 'output' in archi_F.keys():
        return DiscriminatorGP(archi_F['main'], archi_F['output'])
    elif archi_F['module_type'] == 'Sequential':
        return Discriminator(archi_F['conv'], archi_F['fc'], archi_F['dim'], archi_F['depth'], image_size, archi_F.get('inter_dim', None))
    elif archi_F['module_type'] == 'Encoder':
        return Discriminator(encoder=archi_F['main'])

def cast(s):
    '''casts the string s into apprpriate type'''
    def cast_num(n):

        val = None
        try:
            val = int(n)
        except:
            try:
                val = float(n)
            except:

                submodules = n.split('.')
                if len(submodules) > 1: # of type PIL.xxx.kkk
                    assert submodules[0] == 'PIL'
                    val = PIL.__dict__[submodules[1]].__dict__[submodules[2]]


        return val

    is_tuple = len(s.split(',')) > 1

    if is_tuple:
        val = tuple( cast_num(n.strip().lstrip('(').rstrip(')')) for n in s.split(',') if len(n) > 0)
    else:
        if s == 'True':
            val = True
        elif s == 'False':
            val = False
        else:
            val = cast_num(s)
    return val

def parse_layer_args(layer_args_string):


    d = 0  # the parenthesis depth
    buff = []
    key = ''
    kwargs = {}
    args = []
    for idx, c in enumerate(layer_args_string):
        if  c == '(':
            d += 1
        elif c == ')':
            d -= 1
        else:
            if idx == len(layer_args_string)-1:  # last character
                buff.append(c)
            if (d == 0 and c == ',') or idx == len(layer_args_string)-1:
                # end of argument
                buff = ''.join(buff).strip()
                if key:
                    kwargs[key] =  cast(buff)
                else:
                    args.append(cast(buff))
                buff = []
                key = ''
            elif c == '=':
                key = ''.join(buff).strip()
                buff = []
            else:
                buff.append(c)

    return args, kwargs




def parse_log_file(fname, *args):
    '''Parse an entry in a log file (at the beginning of it), of the form
    str_=k
    with k an integer.
    Return k
'''

    ret = {}
    process = Popen(['head', '-n', '40', fname], stdout=PIPE, stderr=PIPE)
    #pr_grep = Popen(['grep', arg], stdin=process.stdout, stdout=PIPE)
    head_lines = process.communicate()[0].decode('utf-8').strip().splitlines()

    all_args = dict()

    def correct_arg_n(arg_n):
        return arg_n.find(',') == -1 and arg_n.find(')') == -1 and arg_n.find(':') == -1

    for line in head_lines:  # for all lines at the top of the file

        kv = line.split('=')  # should be key=value
        #lst = line.split(',')  # some lines can have two elements separated by ','

        if len(kv) == 2 and correct_arg_n(kv[0]):
            all_args[kv[0].strip()] = kv[1].strip()  # only strings


    for arg_n, arg_t in args:

        val_str = all_args.get(arg_n, None)
        if val_str:  # if we found something
            if arg_t is int:
                try:
                    ret[arg_n] = int(val_str)
                except BaseException as e:
                    print(e)
                    continue
            elif arg_t is bool:
                ret[arg_n] = True if val_str == 'True' else False
            elif arg_t is str:
                ret[arg_n] = val_str
            elif arg_t is float:
                try:
                    ret[arg_n] = float(val_str)
                except BaseException as e:
                    print(e)
                    continue
            else:
                print('Not implemented type parse:', arg_t)
                raise NotImplementedError
        else:
            #print('Warning could not find {} in log file'.format(arg_n))
            pass

    return ret

