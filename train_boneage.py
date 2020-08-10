import torch
import numpy as np
import os
import sys
from torchsummary import summary


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

    parser = argparse.ArgumentParser('Training a regression with different number of parameters')
    parser.add_argument('--dataset', '-dat', default='rsna', type=str, help='dataset')
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--output_root', '-oroot', type=str, default='./results/', help='output root for the results')
    parser.add_argument('--name', default='rsna', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4, help='leraning rate')
    parser.add_argument('--save_model', action='store_true', default=False, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=80, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')


    args = parser.parse_args()

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


    imresize = (256, 256)
    #imresize=(64,64)
    #imresize=None
    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                          dataroot=args.dataroot,
                                                             imresize =imresize,
                                                             )
    train_loader, train_size,\
        val_loader, val_size,\
        test_loader, test_size  = utils.get_dataloader( train_dataset, test_dataset, batch_size =args.batch_size, ss_factor=0.8, size_max=args.size_max, pad_collate=(imresize is None))

    #model = models.cnn.CNN(1)
    #model = models.cnn.MaxPoolCNN(1)
    #model = models.cnn.ResizeCNN(imresize)
    #model = models.mlp.MLP(input_dim=imresize[0]*imresize[1])
    feature_extraction = False
    model, input_size = models.pretrained.initialize_model('vgg', feature_extract=feature_extraction)
    modle = models.cnn.CNN3(1)

    imsize = next(iter(train_loader))[0].size()[1:]
    #print('Number of parameters: {}'.format(model.num_parameters(), file=logs))
    print('Number of training samples: {}'.format(len(train_loader.sampler.indices)), file=logs)
    print('Image size (first batch): {}'.format(imsize), file=logs)
    print('Model: {}'.format(str(model)), file=logs)
    model.to(device)
    #summary(model,  [imsize, (1,)])
    #model.apply(models.cnn.init_weights)


    parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]
    #parameters = list(model.parameters())

    optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, betas=(0.95, 0.999), weight_decay=0,
            )
    #optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate)

    #optimizer = torch.optim.SGD(
    #    parameters, lr=0.001, momentum=0.9
    #)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


    stats = {
        'loss_val': [],
        'loss_train': [],
        'epochs': [],
    }
    loss_fn = torch.nn.MSELoss()
    loss_train = 0.

    for epoch in tqdm(range(1, args.nepochs)):

        model.train()

        for idx, (x, is_man, age) in enumerate(train_loader):

            optimizer.zero_grad()
            x = x.to(device)
            is_man = is_man.to(device, dtype)
            age = age.to(device, dtype)
            pred = model(x.expand(-1, 3, -1, -1))#, is_man)
            #pred = model(x, is_man)
            loss = loss_fn(pred, age.view(-1,1))
            loss.backward()
            loss_train = ((idx * loss_train) + loss.item()) / (idx+1)
            optimizer.step()



        stats['loss_train'].append(loss_train)

        model.eval()
        loss_val = 0

        with torch.no_grad():

            for idx, (x, is_man, age) in enumerate(val_loader):

                x = x.to(device, dtype)
                is_man = is_man.to(device, dtype)
                age = age.to(device, dtype)
                pred = model(x.expand(-1, 3, -1, -1))#, is_man)
                #pred = model(x, is_man)
                loss = loss_fn(pred,age.view(-1, 1))
                loss_val = (idx * loss_val + loss.item()) / (idx + 1)  # mean over all val data
                break

        stats['loss_val'].append(loss_val)
        stats['epochs'].append(epoch)
        lr_scheduler.step(loss)

        print('ep {}, train loss {}, val loss {}'.format(
            epoch, stats['loss_train'][-1], stats['loss_val'][-1]),
            file=logs, flush=True)

        if args.save_model and (epoch) % 5 == 0:  # we save every 5 epochs
            checkpoint = {
                'model': model.state_dict(),
                'stats': stats,
                'args': args,
                'optimizer': optimizer.state_dict(),
                'epochs': epoch,
            }
            torch.save(checkpoint, os.path.join(output_path, 'checkpoint.pth'))
