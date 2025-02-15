##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils import create_logger, copy_all_src

from CARPTester import CARPTester as Tester

##########################################################################################

# parameters

env_params = {
    'vertex_size': 50,
    'edge_size': 100,
    'pomo_size': 100,
    'distribution': {
        'data_type': 'smallworld',  # cluster, smallworld, uniform
        'n_cluster': 3,
        'n_cluster_mix': 1,
        'lower': 0.2,
        'upper': 0.8,
        'std': 0.07,
    },
    'load_path': 'D:\CODE\CARP_Distillation_Final\data\carp\carp_smallworld50_10000.pkl',
    'load_raw': None
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        "path": "pretrained/checkpoint-carp-{}.pt".format(env_params['vertex_size']), # directory path of pre-trained model and log files saved.
        'epoch': 'test',  # epoch version of pre-trained model to load.
    },
    'test_episodes': 10000,
    'test_batch_size': 100,
    'test_data_load': {
        'enable': False,
        'filename': ''
    },
}

assert tester_params['test_episodes'] % tester_params['test_batch_size'] == 0, "Number of instances must be divisible by batch size!"


logger_params = {
    'log_file': {
        'desc': 'test_carp{}_{}_epoch{}'.format(env_params['vertex_size'],tester_params['model_load']['path'].split('/')[-2],tester_params['model_load']['epoch']),
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)
    tester.run()

    copy_all_src(tester.result_folder)

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":
    main()
