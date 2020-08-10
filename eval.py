import utils

import torch
import torchvision.utils as vutils
import os
import numpy as np
import math
from subprocess import Popen, PIPE
import argparse
import glob
import matplotlib.pyplot as plt
import shutil
import pdb
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

def main(argv):
    '''find the different models saved in argv.root and write the images to argv.output'''


    names = argv.exp_names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float
    for name in names:  # for the different experiments
        file_lst = [f for f in glob.glob(os.path.join(name, "**", "*.pth"), recursive=True)]
        for f in file_lst:
            dirname, basename = os.path.split(f)
            log_fname = os.path.join(dirname, 'logs.txt')
            try:
                if not os.path.isfile(log_fname):
                    log_fname = os.path.join(os.sep.join(dirname.split(os.sep)[:-1]), 'logs.txt')
                    if not os.path.isfile(log_fname):
                        raise ValueError
                #  rootname = os.path.join(opt.path, 'imgs')
                #args = [
                #        ('batch_size', int),
                #        ('epochs', int),
                #        ('dataset', str),
                #        ('dataroot', str),
                #        ('output_root', str),
                #        ('name', str),
                #    ('learning_rate', float),
                #    ('loss', str),
                #    ('width', int),
                #        ]
                #d_args = utils.parse_log_file(log_fname, *args)
                checkpoint = torch.load(f, map_location=device)
                args = checkpoint['args']
                batch_size = args.batch_size
                dataset = args.dataset
                dataroot = args.dataroot

                archis = utils.parse_archi(log_fname)
                transform = utils.parse_transform(log_fname)

                train_ds, test_ds, num_chs = utils.get_dataset(dataset, dataroot, transform=transform)

                train_loader,train_size,\
                    val_loader, val_size,\
                    test_loader, test_size = utils.get_dataloader(train_ds, test_ds, batch_size, ss_factor=0.1)


                epoch = checkpoint['epochs']
                stats = checkpoint['stats']

                output_path =    os.path.join(os.path.dirname(log_fname), 'eval', f'e-{epoch:03d}')
                if os.path.exists(output_path) and not argv.force:
                    continue
                os.makedirs(output_path, exist_ok=True)

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
                plt.close('all')

                if argv.save_meta:
                    meta_dir, label = os.path.split(os.path.dirname(log_fname))

                    meta_path = os.path.join(meta_dir, 'eval_meta')
                    os.makedirs(meta_path, exist_ok=True)
                    meta_fname = os.path.join(meta_path, 'data.npz')
                    save_meta(meta_fname, stats, args)
                    print('Saved meta data to {}'.format(meta_fname))






            except BaseException as e:
                print(f'Error {str(e)} at line {e.__traceback__.tb_lineno} for file {f}')
                continue

def preprocess_stats(stats):

    '''return a numpy 2d array with correct type'''

    names =  list(stats.keys())
    d = len(stats)
    formats = d* ['f8']
    dtype = dict(names=names, formats=formats)


    Ns = [v.shape[0] for v in stats.values() if v is not None]
    same_size = [n == m for (n, m) in zip(Ns[:-1], Ns[1:])]

    assert all(same_size)
    N = Ns[0]
    #subarray = np.zeros((d,N), dtype=np.float64)

    nan_array = np.nan* np.empty((N,))
    for k, v in stats.items():
        if v is None:
            stats[k] = nan_array

    subarray  = np.array(list(stats.values()), dtype=np.float64).T.copy().view(dtype)

    return subarray



def save_meta(meta_npz_fname, stats, args):
    '''accumulate the obeservation in a meta file'''


    #N_train = train_array.shape[1]
    #N_test = test_array.shape[1]
    loss_train, loss_test = stats['loss_train']['zo'][-1], stats['loss_test']['zo'][-1]
    num_parameters = stats['num_parameters']
    new_entry_train = np.array([(num_parameters, args.__dict__, loss_train)], dtype=[('num_parameters', np.int32 ), ('args', dict), ('loss', np.float32)])
    new_entry_test = np.array([(num_parameters, args.__dict__, loss_test)], dtype=[('num_parameters', np.int32), ('args', dict), ('loss', np.float32)])
    if os.path.isfile(meta_npz_fname):
        meta_data = np.load(meta_npz_fname, allow_pickle=True)

        meta_train = meta_data['train']
        meta_test = meta_data['test']
        meta_nparam = meta_train['num_parameters']
        #names = meta_train['label'].squeeze()
        if num_parameters in meta_nparam:
            # we found a duplicate with the same number of parameters
            idx = np.where(num_parameters == meta_nparam)[0]
            meta_train[idx] = new_entry_train
            meta_test[idx] = new_entry_test
            if len(idx) >=2:
                print('Warning, more than one ducplicate of {} in {}'.format(num_parameters, meta_npz_fname))
        else:

            meta_train = np.vstack((meta_train, new_entry_train))
            meta_test = np.vstack((meta_test, new_entry_test))
    else:
        meta_train = new_entry_train
        meta_test =  new_entry_test
    #meta_train[exp_name] =
    np.savez_compressed(meta_npz_fname, train=meta_train, test=meta_test)

    return





if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='script for generating and saving images')
    parser.add_argument('exp_names',  nargs='*', help='the experiment names to resume')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-f', '--force', default=False, action='store_true', help='force the computation again')
    parser.add_argument('--save_meta', action='store_true', help='saves data for meta comparison')
    #  parser.add_argument('--dry_run', action='store_true', help='dry run mode (do not call sbatch, only copy the files)')
    #  parser.add_argument('--start_idx', type=int, default=1, help='the number of index to start from')
    #  parser.add_argument('--config', help='the default configuration to start from')
    #  parser.add_argument('--batch_template',  default='slurm/scripts/template.sbatch', help='the template sbatch file')
    #  parser.add_argument('script', help='the training script to use')
    #  parser.add_argument('--force_resume', action='store_true', default=False, help='if true, we resume even runs already resumes')
    #  parser.add_argument('--no-touch_resume', action='store_false', dest='touch_resume',  help='if true, we resume even runs already resumes')
    #  iter_parser  = parser.add_mutually_exclusive_group(required=False)
    filter_list = parser.add_mutually_exclusive_group(required=False)
    filter_list.add_argument('--whitelist', help='whitelisting the suffix')
    filter_list.add_argument('--blacklist', help='blacklisting the suffix')
    argv = parser.parse_args()

    main(argv)
