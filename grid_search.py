import utils


import sys
import os
import subprocess
import pdb
from my_utils.utils import MyDict
from collections import OrderedDict
import my_utils.utils as utils
import datetime
import argparse


def bool_tr(key, val):
    return '{}{}'.format('no-' if not val else '', key)

def num_tr(val):
    return '{}{}'.format('m' if val < 0 else '', abs(val))

def tr(key, val):
    if isinstance(val, bool):
        return bool_tr(key, val)
    elif isinstance(val, (int, float)):
        if len(PMT_TR[key]) == 1:
            return '{}{}'.format(PMT_TR[key], num_tr(val))
        else:
            return '{}-{}'.format(PMT_TR[key], num_tr(val))
    else:
        return '{}-{}'.format(PMT_TR[key], val)

def name_suffix_lst(pmt, root='baseline', suffix_key='eps', PMT_DEFAULT={}):
    '''Returns the couple name, suffix from a parameter set'''

    name = '{}'.format(root)
    suffix = ''
    lst = []
    for key, val in pmt.items():
        #  if key == 'eps':  # keep epsilon for suffix
        default = PMT_DEFAULT.get(key, None)  # the default value
        if key == suffix_key:
            suffix = '{}-{}'.format(PMT_TR[key], val)
            lst += ['--{}'.format(key), '{}'.format(val)]
            continue
        if val != default:
            if isinstance(val, bool):
                name += '-' + tr(key, val)
                lst += ['--{}'.format(bool_tr(key, val))]
            elif isinstance(val, int):
                name += '-' + tr(key, val)
                lst += ['--{}'.format(key), '{}'.format(val)]
            elif isinstance(val, float):
                name += '-' + tr(key, val)
                lst += ['--{}'.format(key), '{}'.format(val)]
            elif isinstance(val, str): # such as for the architectures
                if key in PMT_TR.keys():  # we have a short for the name
                    name += '-{}-{}'.format(PMT_TR[key], ''.join(v for v in val if v!='-'))
                else:  # we only use the value
                    name += '-{}'.format(val)
                lst += ['--{}'.format(key), '{}'.format(val)]
            elif isinstance(val, list):
                name += '-{}-{}'.format(PMT_TR[key], ''.join(val))  # of type uFe-DW
                lst += ['--{}'.format(key)] + ['{}'.format(v) for v in val]
            else:
                raise ValueError('{} -- {} not understood'.format(key, val))

    return name, suffix, lst

def main(argv):

    #  global PMT_DEFAULT

    file_baseline =  opt.config
    print('using {}'.format(file_baseline))
    PMT_DEFAULT = utils.load_yaml(file_baseline)
    assert os.path.isfile(file_baseline)
    print(PMT_DEFAULT)
    pmt = PMT_DEFAULT.copy()  # new copy of parameters
    root = os.path.splitext(os.path.basename(file_baseline))[0]  # the name of the config file without extension
    date = datetime.datetime.now().strftime("%m%d%H%M%S")
    batch_template = os.path.abspath('slurm/scripts/template.sbatch')
    #  eps_lst = [1e-2, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
    #
    grid = dict()
    #  suffix_key = 'scaleDistance'
    name_key = opt.name_key if opt.name_key is not None else 'cost'
    suffix_key =  opt.suffix_key if opt.suffix_key is not None else 'epsS'
    #  name_key = 'ncat'
    #  suffix_key = 'scaleDistance'

    #  grid['preprocessG'] = ['none', 'all', 'cont']
    #  grid['cost'] = ['l2']#, 'cos']

    grid['sinkhorn'] = [1]
    #  grid['dztot'] = [27, 128]
    #  grid['nfc'] = [128]
    #  grid['ost'] = ['l1']#, 'l2', 'cos']
    #  pmt['nfc'] = 128
    #  grid['sinkhorn'] = [False, True]
    #  grid['eps'] = {
    #          'entropy' : [0.2, 0.5, 1, 5],  # for entropy regularizer
    #          #  'l2' : [1e-5, 1e-4, 1e-3, 1e-2]  # for no sinkhorn
    #          'l2' : [PMT_DEFAULT['eps']]  # for no sinkhorn
    #  }

    #  grid['nlayers_prG'] = [3, 4]
    #  grid['nlayers_prG'] = [PMT_DEFAULT['nlayers_prG']]
    #  grid['archiD'] = ['swgan__512', 'swgan_512', 'swgan_512__5', 'swgan__10']
    #  grid['archiW'] = ['mlp_5', 'mlp_25', 'mlp_100', 'mlp_128', 'mlp_256']

    #  grid['reg'] = ['l2']#, 'entropy']
    grid['reg'] = ['entropy']#, 'entropy']
    IMG_SIZE = {'infogan': 32, 'dcgan': 64}
    #  grid['reg'] = ['entropy']
    #  grid['scaleDistance'] = [5, 10, 100]#{
            #  True: 5,
            #  False: 1
            #  }
    #  grid['ncat'] = [0, 1, 5]
    #  grid['dzcat'] = {  # the dimension for the categorical variable
    #          0: 0,
    #          1: 10,
    #          5: 2
    #          }

    #
    #  pmt['updateFe'] = [PMT_DEFAULT['updateFe']]
    #  grid['epsS'] = [0, 0.1, 0.5, 1, 2, 5]#0.5, 1, 3, 5, 10, 0]#, 0.1, 1]#, 5]#, 2]#, 2]#, 10]#, 10, 50, 100]
    #  grid['epsW'] = [0, 0.1, 0.5, 1, 2, 5]#, 0.1,1,5,10] #0.5, 1, 3, 5, 10, 0]#, 0.1, 1]#, 5]#, 2]#, 2]#, 10]#, 10, 50, 100]
    grid['eps'] = [0.1]
    #  grid['updateW'] = ['primal', 'dual']
    #  grid['epsW'] = [0.2, 0.5, 1, 5]
    grid['epsS'] = [0, 0.1]
    #  grid['epsCat'] = [0, 0.1, 0.5, 1, 5]
    #  grid['eps'] = [0.1,1,5,10] #0.5, 1, 3, 5, 10, 0]#, 0.1, 1]#, 5]#, 2]#, 2]#, 10]#, 10, 50, 100]
    EPS_MIN = {  #  minimum epsilon weight to use depending on the kind of regularization
            'entropy': 0.1,  # for when entropy
            'l2': 0.01
            }
    #  grid['updateFe'] = ['primal', 'dual']
    #  grid['useFe'] = [False]
    #  grid['epsW'] = [0]
    #  grid['epsS'] = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2]
    #  gir
    #  grid['ncnn'] = [1, 2]
    #  grid['ncnn'] = [1]
    #  grid['ife'] = [0, 5]#, 0, 5]
    grid['iterFe'] = [3, 5, 10]  # update FE every .. generator iteration
    grid['cost'] = ['l2']
    #{  # iter useFE
            #  True: 5,
            #  False: 1
            #  }#, 5]#, 5]
    #  grid['iterFe'] = [PMT_DEFAULT['iterFe']]#, 5]
    #  grid['iterOT'] = [1, 3, 5]
    #  grid['iterOT'] = [3]

    #  pmt['eps'] = 1e-4
    grid['archi'] = [
            #  {
            #  'D': 'infogan__128',
            #  'G': 'infogan'
            #  }
    {
            'D': 'dcgan__128',
            'G': 'dcgan' }
    ]#, 'swgan']#, {'D': 'swgan-dcgan-no-ln', 'G': 'swgan'}]
    #  file_baseline = 'config/0204_baseline.yml' if not len(argv) == 1 else argv
    idx = 0

    #  for reg in grid['reg']:
    #      pmt['reg'] = reg
    #
    #  pmt['scaleDistance'] = 10

    #  for iterOT in grid['iterOT']:
    #      pmt['iterOT'] = iterOT
    #  for sinkhorn in grid['sinkhorn']:
    #      pmt['sinkhorn'] = sinkhorn
    #  for pG in grid['preprocessG']:
        #  pmt['preprocessG'] = pG
    #  #  for ncat in grid['ncat']:
    #  #      pmt['ncat'] = ncat
    #  #      pmt['dzcat'] = grid['dzcat'][ncat]
    #
    #  #  for updateFe in grid['updateFe']:
    #  #      pmt['updateFe'] = updateFe
    #
    #      for useFe in grid['useFe']:
    #          pmt['useFe'] = useFe

    #  for sinkhorn in grid['sinkhorn']:
    #      pmt['sinkhorn'] = sinkhorn
    #      for reg in grid['reg']:
    #          pmt['reg'] = reg
    #          pmt['div_info'] = 'entropy'

    for param in grid[name_key]:
        #  pmt[name_key] = param
        #  reg = pmt['reg']
        #  pmt['eps'] = max(EPS_MIN[reg], pmt['eps'])
        #  useFe = pmt['useFe']
        #  pmt['scaleDistance'] = grid['scaleDistance'][useFe]
        #  pmt['iterFe'] = grid['iterFe'][useFe]
        #  pmt['dzcat'] = grid['dzcat'][pmt['ncat']]

        if pmt['updateFe'] == 'primal' and pmt['useFe'] == False:
            continue
        #  name_job = '{}-{}'.format(root, '-'.join([tr(key, pmt[key]) for key in name_key]]))
        name_job = '{}'.format(root)
        #  name_job = bool_tr('sinkhorn', sinkhorn)
        tmp_batch = []
        idx += 1
        #  name_job += '-{}-{}'.format(suffix_key, idx)
        file_batch = os.path.abspath('slurm/scripts/{}.sbatch'.format(name_job))

        for val in grid[suffix_key]:

            #  pmt[suffix_key] = val
            #  if 0 < pmt['epsS'] < EPS_MIN[reg] or 0 < pmt['epsW'] < EPS_MIN[reg]:
            #      continue

            #  for iterFe in grid['iterFe']:  # suffix loop
            #      pmt['iterFe'] = iterFe

            name, suffix, lst = name_suffix_lst(pmt, root, suffix_key=suffix_key, PMT_DEFAULT=PMT_DEFAULT)
            tmp_batch.append('python {} --config "{}" --name "{}" --suffix "{}" {}\n'.format(opt.train, file_baseline, name, suffix, ' '.join(lst)))
                # end of suffix loop
        subprocess.check_call(['cp', batch_template, file_batch])
        #  subprocess.check_call(['sed', '-i', '--regexp-extended', 's/job-name=.*$/job-name={}/g'.format(''.join(name_job.split('_')[1:])), file_batch])
        subprocess.check_call(['sed', '-i', '--regexp-extended', 's/job-name=.*$/job-name={}/g'.format(name_job), file_batch])

        with open(file_batch, 'at') as _file:
            for exp in tmp_batch:
                _file.write('{}\n'.format(exp))

        print('Calling sbatch {}'.format(file_batch))
        subprocess.Popen(['sbatch', file_batch])

            #  pdb.set_trace()
    #  for eps in eps_lst[1:]:
    #      subprocess.call(['python', 'main.py', '--opt', 'opt/mnist/baseline', '--eps', '{}'.format(eps), '--jname', 'eps{}'.format(eps)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for grid searching parameters')
    parser.add_argument('--start_idx', type=int, default=1, help='the number of index to start from')
    parser.add_argument('--config', help='the default configuration to start from')
    parser.add_argument('--name_key', help='the name key to use')
    parser.add_argument('--suffix_key', help='the suffix key to use')
    parser.add_argument('train', help='the training script to use')
    opt = parser.parse_args()

    main(opt)
