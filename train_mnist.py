import torch
import numpy as np
import os
import sys
from torchsummary import summary
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt


import models

import torch.optim
from datasets.rsna import RSNADataset
import torch
import argparse
import utils

#from torchvision import models, datasets, transforms

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Training a classification with different number of parameters')
    parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--output_root', '-oroot', type=str, default='./results/mnist/200810/', help='output root for the results')
    parser.add_argument('--name', default='baseline', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='leraning rate')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=200, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--width', '-w', type=int, default=1024, help='the width of the hidden layer')
    #parser.add_argument('--loss', '-l', choices=['mse', 'ce'], default='ce', help='the type of loss to use')
    parser.add_argument('--vary_name', nargs='*', default=None, help='the name of the parameter to vary in the name (appended)')
    parser.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser.add_argument('--gd_mode', '-gdm', default='full', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_softmax = parser.add_mutually_exclusive_group(required=False)
    parser_softmax.add_argument('--softmax', action='store_true', help='if set uses the softmax distribution before the MSE loss')
    parser_softmax.add_argument('--no-softmax', dest='softmax', action='store_false', help='use output of last layer direclty')
    parser.set_defaults(softmax=True)



    args = parser.parse_args()



    if args.checkpoint is not None:  # we have some networks weights to continue
        try:
            checkpoint = torch.load(args.checkpoint)  # checkpoint is a dictionnary with different keys
        except RuntimeError as e:
            print('Error loading the checkpoint at {}'.format(e))

    else:
        checkpoint = dict()

    if 'args' in checkpoint.keys():
        args = checkpoint['args']  # restore the previous arguments
    # Logs

    if args.vary_name is not None:
        name = ''
        for field in args.vary_name:
            if field in args:
                arg = args.__dict__[field]
                if isinstance(arg, bool):
                    dirname = f'{field}' if arg else f'no-{field}'
                else:
                    val = str(arg)
                    key = ''.join(c[0] for c in field.split('_'))
                    dirname = f'{key}-{val}'
                name = os.path.join(name, dirname)
        args.name = name if name else args.name

    output_path = os.path.join(args.output_root, args.name)


    os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    if not args.debug:
        logs = open(os.path.join(output_path, 'logs.txt'), 'w')
    else:
        logs = sys.stdout
#     logs = None

    print(os.sep.join((os.path.abspath(__file__).split(os.sep)[-2:])), file=logs)  # folder + name of the script
    print('device= {}, num of gpus= {}'.format(device, num_gpus), file=logs)
    print('dtype= {}'.format(dtype), file=logs)

    for k, v in vars(args).items():
        print("%s= %s" % (k, v), file=logs, flush=True)


    #imresize = (256, 256)
    #imresize=(64,64)
    imresize=None
    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                          dataroot=args.dataroot,
                                                             imresize =imresize,
                                                             )
    train_loader, size_train,\
        val_loader, size_val,\
        test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset, batch_size =args.batch_size, ss_factor=1, size_max=args.size_max, collate_fn=None)

    #model = models.cnn.CNN(1)

    num_classes = 10
    imsize = next(iter(train_loader))[0].size()[1:]
    input_dim = imsize[0]*imsize[1]*imsize[2]
    model = models.classifiers.FCNN(input_dim=input_dim, num_classes=num_classes, width=args.width)

    num_parameters = utils.num_parameters(model)
    num_samples_train = size_train
    num_samples_val = size_val
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    print('Input dimension: '.format(input_dim), file=logs)
    print('Model: {}'.format(str(model)), file=logs)
    model.to(device)
    #summary(model,  [imsize, (1,)])
    #model.apply(models.cnn.init_weights)

    if 'model' in checkpoint.keys():
        try:
            model.load_state_dict(checkpoint['model'])
            model.train()
        except RuntimeError as e:
            print("Can't load mode (error {})".format(e))



    #parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]
    parameters = list(model.parameters())

    #optimizer = torch.optim.AdamW(
    #        parameters, lr=args.learning_rate, betas=(0.95, 0.999), weight_decay=0,
    #        )
    #optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate)

    optimizer = torch.optim.SGD(
        parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.9
    )
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    if 'optimizer' in checkpoint.keys():

        try:
            optimizer.load_state_key(checkpoint['optimizer'])
        except RuntimeError as e:
            print("Can't load model (error {})".format(e))

    if 'lr_scheduler' in checkpoint.keys():

        try:
            lr_scheduler.load_state_key(checkpoint['lr_scheduler'])
        except RuntimeError as e:
            print("Can't load model (error {})".format(e))

    stats = {
        'loss_test': defaultdict(list),
        'loss_train': defaultdict(list),
        'epochs': [],
        'num_parameters': num_parameters,
        'num_samples_train': num_samples_train,
        'lr': [],
    }

    if 'stats' in checkpoint.keys():
        stats.update(checkpoint['stats'])

    classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes

    def zero_one_loss(x, targets):
        return  (x!=y).float().mean()

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    start_epoch = 0
    if 'epochs' in checkpoint.keys():
        start_epoch = checkpoint['epochs']

    for epoch in tqdm(range(start_epoch+1, start_epoch+args.nepochs+1)):

        model.train()
        loss_train = np.zeros(3)
        # 0: cel loss
        # 1: 0-1 loss
        # 2: mse loss
        losses = np.zeros(3)

        if args.gd_mode == 'full':
            optimizer.zero_grad()

        for idx, (x, y) in enumerate(train_loader):

            if args.gd_mode == 'stochastic':
                optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            #age = age.to(device, dtype)
            out = model(x)#, is_man)
            out_exp = out.exp()
            S = out_exp.sum(dim=1, keepdim=True)
            pred = out_exp / S

            y_onehot = (y.view(-1, 1) ==  classes).float()
            if args.softmax:
                loss = mse_loss(pred, y_onehot)
            else:
                loss = mse_loss(out, y_onehot)

            losses[0] = loss.item()
            losses[1] = zero_one_loss(out.argmax(dim=1), y).item()
            losses[2] =  ce_loss(out, y).item()
            #pred = model(x, is_man)
            loss.backward()
            loss_train = ((idx * loss_train) + losses) / (idx+1)
            if args.gd_mode == 'stochastic':
                optimizer.step()


        if args.gd_mode == 'full':
            optimizer.step()



        stats['loss_train']['mse'].append(loss_train[0])
        stats['loss_train']['ce'].append(loss_train[2])
        stats['loss_train']['zo'].append(loss_train[1])

        model.eval()
        loss_test = np.zeros(3)

        with torch.no_grad():

            for idx, (x, y) in enumerate(test_loader):

                x = x.to(device, dtype)
                y = y.to(device, dtype=torch.long)
                out = model(x)#, is_man)
                out_exp = out.exp()
                S = out_exp.sum(dim=1, keepdim=True)
                pred = out_exp / S

                #pred = model(x, is_man)
                #loss = loss_fn(pred,y)
                y_onehot = (y.view(-1, 1) ==  classes).float()
                losses[0] = zero_one_loss(out.argmax(dim=1), y).item()
                losses[1] = ce_loss(out, y).item()
                if args.softmax:
                    losses[2] = mse_loss(pred, y_onehot).item()
                else:
                    losses[2] = mse_loss(out, y_onehot).item()
                #losses[2] = mse_loss(pred, y_onehot).item()

                loss_test = (idx * loss_test + losses) / (idx + 1)  # mean over all test data

        stats['loss_test']['zo'].append(loss_test[0])
        stats['loss_test']['ce'].append(loss_test[1])
        stats['loss_test']['mse'].append(loss_test[2])
        stats['epochs'].append(epoch)
        #lr_scheduler.step(loss)
        lr_scheduler.step()
        stats['lr'].append(lr_scheduler.get_last_lr())


        print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g})'.format(
            epoch, stats['loss_train']['mse'][-1], stats['loss_train']['zo'][-1],
            stats['loss_test']['mse'][-1], stats['loss_test']['zo'][-1]),
            file=logs, flush=True)

        fig = plt.figure()
        plt.plot(stats['epochs'], stats['loss_train']['mse'], label='Train loss')
        plt.plot(stats['epochs'], stats['loss_test']['mse'], label='Test loss')
        plt.legend()
        plt.savefig(fname=os.path.join(output_path, 'mse_loss.pdf'))

        fig = plt.figure()
        plt.plot(stats['epochs'], stats['loss_train']['zo'], label='Train loss')
        plt.plot(stats['epochs'], stats['loss_test']['zo'], label='Test loss')
        plt.legend()
        plt.savefig(fname=os.path.join(output_path, 'zero_one_loss.pdf'))

        fig = plt.figure()
        plt.plot(stats['epochs'], stats['loss_train']['ce'], label='Train loss')
        plt.plot(stats['epochs'], stats['loss_test']['ce'], label='Test loss')
        plt.legend()
        plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss.pdf'))

        fig=plt.figure()
        plt.plot(stats['epochs'], stats['lr'], label='lr')
        plt.legend()
        plt.savefig(fname=os.path.join(output_path, 'lr.pdf'))

        plt.close('all')

        if args.save_model and (epoch) % 5 == 0 or (epoch==start_epoch+args.nepochs):  # we save every 5 epochs
            checkpoint = {
                'model': model.state_dict(),
                'stats': stats,
                'args': args,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epochs': epoch,
            }
            torch.save(checkpoint, os.path.join(output_path, 'checkpoint.pth'))


