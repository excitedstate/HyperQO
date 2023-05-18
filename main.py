"""
    this file used to run the startup program
"""
import random
import sys
import src.config as config
from src.hints import Hinter
from src.mcts import MCTSHinterSearch
from src.basic import load_json
from src.net import TreeNet
from src.encoding import SQLEncoder, TreeBuilder, value_extractor
from src.tree_lstm import SPINN
from torch.nn import init


def train(queries, hinter):
    sys.stdout = open(config.LOG_FILE_NAME, "w")
    print(len(queries))
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s_pg = 0
    s_hinter = 0
    for epoch in range(1):
        for idx, x in enumerate(queries[:]):
            print('----', idx, '-----')
            pg_plan_time, pg_latency, mcts_time, hinter_plan_time, MPHE_time, hinter_latency, actual_plans, actual_time = hinter.hinterRun(
                x[0])
            pg_latency /= 1000
            hinter_latency /= 1000
            pg_plan_time /= 1000
            hinter_plan_time /= 1000
            print('pg plan:', pg_plan_time, 'pg run:', pg_latency)
            s1 += pg_plan_time
            print('mcts:', mcts_time, 'plan gen:', hinter_plan_time, 'MPHE:', MPHE_time, 'hinter latency:',
                  hinter_latency)
            s2 += mcts_time
            s3 += hinter_plan_time
            s4 += MPHE_time
            s_pg += pg_latency
            s_hinter += sum(actual_time) / 1000
            # print()
            print([actual_plans, actual_time])
            print("%.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (s1, s2, s3, s4, s_pg, s_hinter, s_hinter / s_pg))

            sys.stdout.flush()


def main():
    random.seed(113)
    queries = load_json(config.INPUT_WORKLOAD_NAME)

    tree_builder = TreeBuilder()
    sql2vec = SQLEncoder()
    """
        NOTE: BING 2023/5/18 下午4:27 
        head_num: int
        input_size: int
        hidden_size: int
        table_num: int
        sql_size: int
    """
    value_network = SPINN(head_num=config.NET_HEAD_NUM, input_size=7 + 2, hidden_size=config.NET_HIDDEN_SIZE,
                          table_num=50,
                          sql_size=40 * 40 + config.MAX_COLUMN_ID).to(config.DEVICE_NAME)
    for name, param in value_network.named_parameters():
        if len(param.shape) == 2:
            init.xavier_normal(param)
        else:
            init.uniform(param)

    net = TreeNet(tree_builder=tree_builder, value_network=value_network)

    mcts_searcher = MCTSHinterSearch()
    hinter = Hinter(model=net, sql2vec=sql2vec, value_extractor=value_extractor, mcts_searcher=mcts_searcher)

    train(queries, hinter)


if __name__ == '__main__':
    main()