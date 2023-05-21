import logging
import time
import typing
from collections import namedtuple

import numpy as np
import torchfold
import torch

import src.config as config
from src.basic import PostgresDB
from src.encoding import SQLEncoder, ValueExtractor

from src.knn import KNN
from src.mcts import MCTSHinterSearch
from src.net import TreeNet


class Timer:
    def __init__(self, ):
        self.time_eval: typing.Optional[typing.Callable[[], float]] = time.time
        self.start_time_record = dict()

    def reset(self, event: str):
        """
            记录事件 event 最后一次发生的时间
        @param event:
        @return: 
        """
        self.start_time_record[event] = self.time_eval()

    def record(self, event: str):
        """
            记录事件 event 最后一次发生时间和当前时间的差值
        @param event:
        @return:
        """
        return self.time_eval() - self.start_time_record[event]


class TimerRecord:
    TIMER_RECORD_SAMPLE = namedtuple('TimerRecord',
                                     ['pg_planning_time',
                                      'pg_running_time',
                                      'mcts_time',
                                      'hinter_planning_time',
                                      'MHPE_time',
                                      'hinter_runtime',
                                      'chosen_plan',
                                      'hinter_time'])

    def __init__(self):
        self.pg_planning_time_list = []
        self.pg_running_time_list = []  # default pg running time
        self.mcts_time_list = []  # time for mcts
        self.hinter_planning_time_list = []  # chosen hinter running time,include the timeout
        self.mhpe_time_list = []
        self.hinter_runtime_list = []
        self.chosen_plan_list = []  # eg((leading ,pg))
        self.hinter_time_list = []  # final plan((eg [(leading,),(leading,pg),...]))

    def __getitem__(self, item) -> TIMER_RECORD_SAMPLE:
        return TimerRecord.TIMER_RECORD_SAMPLE(self.pg_planning_time_list[item], self.pg_running_time_list[item],
                                               self.mcts_time_list[item],
                                               self.hinter_planning_time_list[item], self.mhpe_time_list[item],
                                               self.hinter_runtime_list[item],
                                               self.chosen_plan_list[item], self.hinter_time_list[item])


class HintGenerator:
    def __init__(self, tree_net: TreeNet,
                 sql2vec: SQLEncoder,
                 value_extractor: ValueExtractor,
                 mcts_searcher: typing.Optional[MCTSHinterSearch] = None):
        self.knn = KNN(10)
        # # models
        self.tree_net = tree_net  # encode query plan
        self.sql2vec = sql2vec  # encode sql query
        self.value_extractor = value_extractor  # # encode and decode
        self.mcts_searcher = mcts_searcher  # # mcts to get hint

        self.db = PostgresDB.default()

        self.hinter_count = 0
        self.timer = Timer()
        self.timer_record = TimerRecord()

    def hinter_run(self, sql: str):
        """

        @param sql:
        @return:
        """
        self.hinter_count += 1
        plan_json_pg = self.db.get_cost_plan(sql)

        samples_plan_with_time = []
        mask = (torch.rand(1, config.NET_HEAD_NUM, device=config.DEVICE_NAME) < 0.9).long()

        if config.COST_TEST_FOR_DEBUG:
            self.timer_record.pg_running_time_list.append(self.db.get_cost(sql)[0])
            self.timer_record.pg_planning_time_list.append(self.db.get_cost_plan(sql)['Planning Time'])
        else:
            self.timer_record.pg_running_time_list.append(self.db.get_analyse_plan(sql)['Plan']['Actual Total Time'])
            self.timer_record.pg_planning_time_list.append(self.db.get_analyse_plan(sql)['Planning Time'])

        sql_vec, alias = self.sql2vec.encode(sql)
        plan_jsons = [plan_json_pg]
        plan_times = self.predict_with_uncertainty_batch(plan_jsons=plan_jsons, sql_vec=sql_vec)

        # # list[( plan_time, leading, leading_utility )], 0 就是最好的
        chosen_leading_pair = self.find_base_hint(plan_json_pg=plan_json_pg, alias=alias, sql_vec=sql_vec, sql=sql)
        knn_plan = abs(self.knn.k_neighbours_sample(plan_times[0]))

        if chosen_leading_pair[0][0] < plan_times[0][0] and abs(
                knn_plan) < config.THRESHOLD and self.value_extractor.decode(plan_times[0][0]) > 100:
            # # 优化不太行
            max_time_out = min(int(self.value_extractor.decode(chosen_leading_pair[0][0]) * 3), config.MAX_TIME_OUT)
            if config.COST_TEST_FOR_DEBUG:
                # # todo: bug here
                leading_time_flag = self.db.get_cost(sql=chosen_leading_pair[1] + sql)
                self.timer_record.hinter_runtime_list.append(leading_time_flag[0])
                self.timer_record.hinter_planning_time_list.append(
                    self.db.get_cost_plan(sql=chosen_leading_pair[1] + sql)['Planning Time'])
            else:
                plan_json = self.db.get_analyse_plan(sql=chosen_leading_pair[1] + sql)
                leading_time_flag = (plan_json['Plan']['Actual Total Time'], plan_json['timeout'])
                self.timer_record.hinter_runtime_list.append(leading_time_flag[0])
                self.timer_record.hinter_planning_time_list.append(plan_json['Planning Time'])

            self.knn.insert_values((chosen_leading_pair[0],
                                    self.value_extractor.encode(leading_time_flag[0]) - chosen_leading_pair[0][0]))

            samples_plan_with_time.append([
                self.db.get_cost_plan(sql=chosen_leading_pair[1] + sql, timeout=max_time_out),
                leading_time_flag[0],
                mask
            ])

            if leading_time_flag[1]:
                if config.COST_TEST_FOR_DEBUG:
                    pg_time_flag = self.db.get_cost(sql=sql)
                else:
                    pg_time_flag = self.db.get_latency(sql=sql, timeout=300 * 1000)
                self.knn.insert_values((plan_times[0], self.value_extractor.encode(pg_time_flag[0]) - plan_times[0][0]))
                if samples_plan_with_time[0][1] > pg_time_flag[0] * 1.8:
                    samples_plan_with_time[0][1] = pg_time_flag[0] * 1.8
                    samples_plan_with_time.append([plan_json_pg, pg_time_flag[0], mask])
                else:
                    samples_plan_with_time[0] = [plan_json_pg, pg_time_flag[0], mask]

                self.timer_record.hinter_time_list.append(
                    [max_time_out, self.db.get_latency(sql=sql, timeout=300 * 1000)[0]])
                self.timer_record.chosen_plan_list.append([chosen_leading_pair[1], 'PG'])
            else:
                self.timer_record.hinter_time_list.append([leading_time_flag[0]])
                self.timer_record.chosen_plan_list.append([chosen_leading_pair[1]])
        else:
            # # 优化可以
            if config.COST_TEST_FOR_DEBUG:
                pg_time_flag = self.db.get_cost(sql=sql)
                self.timer_record.hinter_runtime_list.append(pg_time_flag[0])
                self.timer_record.hinter_planning_time_list.append(self.db.get_cost_plan(sql)['Planning Time'])
            else:
                pg_time_flag = self.db.get_latency(sql=sql, timeout=300 * 1000)
                self.timer_record.hinter_runtime_list.append(pg_time_flag[0])

                self.timer_record.hinter_planning_time_list.append(self.db.get_analyse_plan(sql=sql)['Planning Time'])

            self.knn.insert_values((plan_times[0], self.value_extractor.encode(pg_time_flag[0]) - plan_times[0][0]))
            samples_plan_with_time.append([plan_json_pg, pg_time_flag[0], mask])

            self.timer_record.hinter_time_list.append([pg_time_flag[0]])
            self.timer_record.chosen_plan_list.append(['PG'])

        for sample in samples_plan_with_time:
            # # 选出计划, 更新tree_net和mcts_searcher
            try:
                target_value = self.value_extractor.encode(sample[1])
                self.tree_net.train(plan_json=sample[0], sql_vec=sql_vec, target_value=target_value, mask=mask,
                                    optimize=True)
                self.mcts_searcher.train(tree_feature=self.tree_net.tree_builder.plan_to_feature_tree(sample[0]),
                                         sql_vec=sql_vec, target_value=sample[1], alias_set=alias)
            except Exception as e:
                logging.error(f"{e=}")

        if self.hinter_count < 1000 or self.hinter_count % 10 == 0:
            # # 记录loss
            loss = self.tree_net.optimize()[0]
            loss1 = self.mcts_searcher.optimize()
            if self.hinter_count < 1000:
                loss = self.tree_net.optimize()[0]
                loss1 = self.mcts_searcher.optimize()
            if loss > 3:
                loss = self.tree_net.optimize()[0]
                loss1 = self.mcts_searcher.optimize()
            if loss > 3:
                loss = self.tree_net.optimize()[0]
                loss1 = self.mcts_searcher.optimize()
            logging.debug(f"{loss=}, {loss1=}")

        # # 这些指标集记录的时间次数应该是一样的
        assert len({len(self.timer_record.hinter_runtime_list), len(self.timer_record.pg_running_time_list),
                    len(self.timer_record.mcts_time_list),
                    len(self.timer_record.hinter_planning_time_list), len(self.timer_record.mhpe_time_list),
                    len(self.timer_record.hinter_runtime_list),
                    len(self.timer_record.chosen_plan_list), len(self.timer_record.hinter_time_list)}) == 1
        return self.timer_record[-1]

    def find_base_hint(self, plan_json_pg: dict, alias: set[str], sql_vec: np.ndarray, sql: str):
        """

        @param plan_json_pg: has two key: Plan and Planning Time
        @param alias:
        @param sql_vec: &alias, from sql_encoder.encode()
        @param sql: origin sql
        @return:
        """
        # # get ids for alias, joins...
        alias_id = [self.sql2vec.aliasname2id[a] for a in alias]
        id_joins_with_predicate = [(self.sql2vec.aliasname2id[p[0]], self.sql2vec.aliasname2id[p[1]]) for p in
                                   self.sql2vec.join_list_with_predicate]
        id_joins = [(self.sql2vec.aliasname2id[p[0]], self.sql2vec.aliasname2id[p[1]]) for p in self.sql2vec.join_list]

        leading_length = config.LEADING_LENGTH

        if leading_length == -1:
            leading_length = len(alias)
        if leading_length > len(alias):
            leading_length = len(alias)
        # # 1. 获取hints
        self.timer.reset('mcts_time_list')
        # # find candidate hints
        join_list_with_predicate = self.mcts_searcher.find_candidate_hints(len(alias), sql_vec, id_joins,
                                                                           id_joins_with_predicate, alias_id,
                                                                           depth=leading_length)
        self.timer_record.mcts_time_list.append(self.timer.record('mcts_time_list'))

        leading_list = []
        plan_jsons = []
        leadings_utility_list = []
        # # ... 获取 leadings
        for join in join_list_with_predicate:
            leading_list.append(
                '/*+Leading(' + " ".join([self.sql2vec.id2aliasname[x] for x in join[0][:leading_length]]) + ')*/')

            leadings_utility_list.append(join[1])
            # # 最好用连接池
            plan_jsons.append(self.db.get_cost_plan(leading_list[-1] + sql))
        # # 最后加入 cost - based plan
        plan_jsons.extend([plan_json_pg])
        # # 2. 获取不确定性预测结果
        self.timer.reset('MHPE_time_list')
        plan_times = self.predict_with_uncertainty_batch(plan_jsons=plan_jsons, sql_vec=sql_vec)
        self.timer_record.mhpe_time_list.append(self.timer.record('MHPE_time_list'))
        # # 3. 排序, 选出最好的几个结果 list[( plan_time, leading, leading_utility )]
        chosen_leading_pair = sorted(zip(plan_times[:config.MAX_HINT_COUNT], leading_list, leadings_utility_list),
                                     key=lambda x: x[0][0] + self.knn.k_neighbours_sample(x[0]))[0]
        return chosen_leading_pair

    def predict_with_uncertainty_batch(self, plan_jsons: list[dict], sql_vec: np.ndarray):
        """
            基本就是按照公式 直接算
        @param plan_jsons:
        @param sql_vec:
        @return:
        """
        # # to tensors
        sql_feature = self.tree_net.value_network.sql_feature(sql_vec)

        fold = torchfold.Fold(cuda=False)
        multi_list = []
        for plan_json in plan_jsons:
            try:
                tree_feature = self.tree_net.tree_builder.plan_to_feature_tree(plan_json)
                multi_value = self.tree_net.plan_to_value_fold(tree_feature=tree_feature, sql_feature=sql_feature,
                                                               fold=fold)
                multi_list.append(multi_value)
            except Exception as e:
                print(e)
        multi_value = fold.apply(self.tree_net.value_network, [multi_list])[0]
        mean, variance = self.tree_net.mean_and_variance(multi_value=multi_value[:, :config.NET_HEAD_NUM])
        v2 = torch.exp(multi_value[:, config.NET_HEAD_NUM] * config.VAR_WEIGHT).data.reshape(-1)
        if isinstance(mean, float):
            mean_item = [mean]
        else:
            mean_item = [x.item() for x in mean]
        if isinstance(variance, float):
            variance_item = [variance]
        else:
            variance_item = [x.item() for x in variance]
        # variance_item = [x.item() for x in variance]
        if isinstance(v2, float):
            v2_item = [v2]
        else:
            v2_item = [x.item() for x in v2]
        # v2_item = [x.item() for x in v2]
        res = list(zip(mean_item, variance_item, v2_item))
        return res
