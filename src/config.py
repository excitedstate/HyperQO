import math
import os.path

import torch

PG_HOST = "127.0.0.1"
PG_PORT = 5432
PG_USER = 'bing'
PG_PASSWORD = "bing"
PG_DATABASE = 'imdbload'

NET_USE_GPU = False  # # unused

NET_HEAD_NUM = 10
NET_INPUT_SIZE = 9
NET_HIDDEN_SIZE = 64
NET_BATCH_SIZE = 256

DEVICE_NAME = "cpu"
CPU_DEVICE_NAME = "cpu"

VAR_WEIGHT = 0.00  # for au, 0:disable,0.01:enable

MAX_COLUMN_ID = 100
MAX_ALIAS_ID = 40

MAX_HINT_COUNT = 20
MAX_TIME_OUT = 120 * 1000
THRESHOLD = math.log(3) / math.log(MAX_TIME_OUT)
LEADING_LENGTH = 2
HINT_TRY_TIMES = 3

MEM_SIZE = 2000

MCTS_V = 1.1
MCTS_INPUT_SIZE = MAX_ALIAS_ID * MAX_ALIAS_ID + MAX_COLUMN_ID

SEARCH_SIZE = 4
U_FACTOR = 0.0

COST_TEST_FOR_DEBUG = False

PROJECT_ROOT = '/home/bing/Projects/PythonProjects/HyperQO/'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
LOGS_PATH = os.path.join(DATA_PATH, 'logs')
SAVE_MODELS_PATH = os.path.join(DATA_PATH, 'model')

WORKLOADS_PATH = os.path.join(DATA_PATH, 'workload')
LOG_FILE_NAME = os.path.join(LOGS_PATH, "log_c3_h64_s4_t3.log")
LOG_LATENCY_FILE_NAME = os.path.join(LOGS_PATH, "latency_record.log")
INPUT_WORKLOAD_NAME = os.path.join(WORKLOADS_PATH, "JOB_static.json")

OFFSET = 20

# # 一共40个别名
ID2ALIAS_NAME = {0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt',
                 10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt',
                 18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3',
                 26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1',
                 34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'}
ALIAS_NAME2ID = {'start': 0, 'chn': 1, 'ci': 2, 'cn': 3, 'ct': 4, 'mc': 5, 'rt': 6, 't': 7, 'k': 8, 'lt': 9,
                 'mk': 10, 'ml': 11, 'it1': 12, 'it2': 13, 'mi': 14, 'mi_idx': 15, 'it': 16, 'kt': 17,
                 'miidx': 18, 'at': 19, 'an': 20, 'n': 21, 'cc': 22, 'cct1': 23, 'cct2': 24, 'it3': 25,
                 'pi': 26, 't1': 27, 't2': 28, 'cn1': 29, 'cn2': 30, 'kt1': 31, 'kt2': 32, 'mc1': 33,
                 'mc2': 34, 'mi_idx1': 35, 'mi_idx2': 36, 'an1': 37, 'n1': 38, 'a1': 39}


def get_device():
    if NET_USE_GPU:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(DEVICE_NAME)
