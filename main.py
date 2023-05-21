"""
    this file used to run the startup program
"""
import dataclasses
import random
import sys
import src.config as config
from src.hints import HintGenerator
from src.mcts import MCTSHinterSearch
from src.basic import load_json
from src.net import TreeNet
from src.encoding import SQLEncoder, TreeBuilder
from src.tree_lstm import SPINN


def train(queries: list[str, str, list], hinter: HintGenerator):
    sys.stdout = open(config.LOG_FILE_NAME, "w")
    print(len(queries))
    sum_pg_plan_time = 0
    sum_mcts_time = 0
    sum_hinter_plan_time = 0
    sum_mphe_time = 0
    sum_pg = 0
    sum_hinter = 0
    for epoch in range(1):
        for idx, (sql, _, _) in enumerate(queries[:]):
            (pg_plan_time, pg_latency, mcts_time, hinter_plan_time, mphe_time, hinter_latency,
             actual_plans, actual_time) = hinter.hinter_run(sql)
            pg_latency /= 1000
            hinter_latency /= 1000
            pg_plan_time /= 1000
            hinter_plan_time /= 1000
            sum_pg_plan_time += pg_plan_time
            sum_mcts_time += mcts_time
            sum_hinter_plan_time += hinter_plan_time
            sum_mphe_time += mphe_time
            sum_pg += pg_latency
            sum_hinter += sum(actual_time) / 1000

            print(f'---------------------------{idx}---------------------------')
            print(f'pg plan: {pg_plan_time}, pg run: {pg_latency}')
            print(f'mcts: {mcts_time}, plan gen: {hinter_plan_time}, '
                  f'MPHE: {mphe_time}, hinter latency: {hinter_latency}')
            print(f"{actual_plans=}, {actual_time=}")
            print(f"now: {sum_pg_plan_time=:.4f} {sum_mcts_time=:.4f} {sum_hinter_plan_time=:.4f} "
                  f"{sum_mphe_time=:.4f} {sum_pg=:.4f} {sum_hinter=:.4f} {sum_hinter/sum_pg=:.4f}")

            sys.stdout.flush()


def main():
    random.seed(113)

    tree_builder = TreeBuilder()
    value_network = SPINN(head_num=config.NET_HEAD_NUM,
                          input_size=7 + 2,
                          hidden_size=config.NET_HIDDEN_SIZE,
                          table_num=50,
                          sql_size=40 * 40 + config.MAX_COLUMN_ID).to(config.DEVICE_NAME)

    tree_net = TreeNet(tree_builder=tree_builder, value_network=value_network)
    mcts_searcher = MCTSHinterSearch()

    train(load_json(config.INPUT_WORKLOAD_NAME), hinter=HintGenerator(tree_net=tree_net,
                                                                      sql2vec=SQLEncoder(),
                                                                      value_extractor=tree_builder.value_extractor,
                                                                      mcts_searcher=mcts_searcher))


if __name__ == '__main__':
    main()
