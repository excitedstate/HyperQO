import torch
import math

datafile = 'JOBqueries.workload'  # # unused
schemaFile = "schema.sql"  # # unused
database = 'imdbload'
user = 'bing'
password = "bing"
dataset = 'JOB'  # # unused
userName = user
usegpu = False  # # unused
head_num = 10
input_size = 9
hidden_size = 64
batch_size = 256
ip = "127.0.0.1"
port = 5432
device = "cpu"
cpudevice = "cpu"
var_weight = 0.00  # for au, 0:disable,0.01:enable
max_column = 100
max_alias_num = 40
cost_test_for_debug = False
max_hint_num = 20
max_time_out = 120 * 1000
threshold = math.log(3) / math.log(max_time_out)
leading_length = 2
try_hint_num = 3
mem_size = 2000
mcts_v = 1.1
mcts_input_size = max_alias_num * max_alias_num + max_column
searchFactor = 4
U_factor = 0.0
log_file = 'log_c3_h64_s4_t3.txt'
latency_file = 'latency_record.txt'
queries_file = 'data/workload/JOB_static.json'
id2aliasname = {0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt',
                10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt',
                18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3',
                26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1',
                34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'}
aliasname2id = {'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32,
                'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8,
                'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19,
                'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12,
                'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33}
modelpath = 'model/'
offset = 20


def get_device():
    if usegpu:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)
