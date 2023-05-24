import dataclasses
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
                                      'MHPE_time',  # # multi-head-prediction-estimator
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


class HyperQO:
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

        self.invoke_count = 0
        self.timer = Timer()
        self.timer_record = TimerRecord()

    def optimize(self, sql: str):
        """
            for each sql query, run the hint generator
        @param sql:
        @return:
        """
        self.invoke_count += 1

        # # 这三行是为了调试
        analyse_plan_res = self.db.get_analyse_plan(sql)
        self.timer_record.pg_running_time_list.append(analyse_plan_res['Plan']['Actual Total Time'])
        self.timer_record.pg_planning_time_list.append(analyse_plan_res['Planning Time'])

        # # 1. get the cost-based plan from pg
        plan_json_pg = self.db.get_cost_plan(sql)

        # # 2. encode sql and get related table
        sql_vec, alias = self.sql2vec.encode(sql)

        chosen_leading_pairs = self.find_base_hint(plan_json_pg=plan_json_pg,
                                                   alias=alias,
                                                   sql_vec=sql_vec,
                                                   sql=sql)

        mask = (torch.rand(1, config.NET_HEAD_NUM, device=config.DEVICE_NAME) < 0.9).long()

        samples_plan_with_time = self.adaptive(sql_vec, chosen_leading_pairs[0], sql, mask, plan_json_pg)

        self.batch_train(samples_plan_with_time, sql_vec, mask, alias)

        return *self.timer_record[-1], chosen_leading_pairs

    def adaptive(self, sql_vec, chosen_leading_pair, sql: str, mask, plan_json_pg):
        """
            这个函数是关键
        @param sql_vec:
        @param chosen_leading_pair:
        @param sql:
        @param mask:
        @param plan_json_pg:
        @return:
        """
        plan_times = self.predict_with_uncertainty_batch(plan_jsons=[plan_json_pg], sql_vec=sql_vec)
        samples_plan_with_time = list()
        knn_plan = abs(self.knn.k_neighbours_sample(plan_times[0]))
        # # 0 0 就是 mean time
        if chosen_leading_pair[0][0] < plan_times[0][0] and knn_plan < config.THRESHOLD and self.value_extractor.decode(
                plan_times[0][0]) > 100:
            # # 最好的 满足 要求,
            max_time_out = min(int(self.value_extractor.decode(chosen_leading_pair[0][0]) * 3), config.MAX_TIME_OUT)
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
                # # 超时了
                pg_execute_time = self.db.get_latency(sql=sql, timeout=300 * 1000)
                self.knn.insert_values(
                    (plan_times[0], self.value_extractor.encode(pg_execute_time[0]) - plan_times[0][0]))
                if samples_plan_with_time[0][1] > pg_execute_time[0] * 1.8:
                    samples_plan_with_time[0][1] = pg_execute_time[0] * 1.8
                    samples_plan_with_time.append([plan_json_pg, pg_execute_time[0], mask])
                else:
                    samples_plan_with_time[0] = [plan_json_pg, pg_execute_time[0], mask]
                # # 用 PG
                self.timer_record.hinter_time_list.append(
                    [max_time_out, self.db.get_latency(sql=sql, timeout=300 * 1000)[0]])
                self.timer_record.chosen_plan_list.append([chosen_leading_pair[1], 'PG'])
            else:
                # # 用 hint
                self.timer_record.hinter_time_list.append([leading_time_flag[0]])
                self.timer_record.chosen_plan_list.append([chosen_leading_pair[1]])
        else:
            # # 用pg
            # # 获取原SQL的执行时间和计划时间
            pg_execute_time, _ = self.db.get_latency(sql=sql, timeout=300 * 1000)
            self.timer_record.hinter_runtime_list.append(pg_execute_time)
            self.timer_record.hinter_planning_time_list.append(self.db.get_analyse_plan(sql=sql)['Planning Time'])

            self.knn.insert_values((plan_times[0], self.value_extractor.encode(pg_execute_time) - plan_times[0][0]))
            samples_plan_with_time.append([plan_json_pg, pg_execute_time, mask])

            self.timer_record.hinter_time_list.append([pg_execute_time])
            self.timer_record.chosen_plan_list.append(['PG'])
        return samples_plan_with_time

    def batch_train(self, samples_plan_with_time, sql_vec, mask, alias):
        for sample in samples_plan_with_time:
            # # 增量训练 tree_net 和 mcts_searcher
            try:
                target_value = self.value_extractor.encode(sample[1])
                self.tree_net.train(plan_json=sample[0], sql_vec=sql_vec, target_value=target_value, mask=mask,
                                    optimize=True)
                self.mcts_searcher.train(tree_feature=self.tree_net.tree_builder.plan_to_feature_tree(sample[0]),
                                         sql_vec=sql_vec, target_value=sample[1], alias_set=alias)
            except Exception as e:
                logging.error(f"{e=}")

        if self.invoke_count < 1000 or self.invoke_count % 10 == 0:
            # # 利用保存的数据训练tree_net和mcts_searcher
            loss_tree_net = self.tree_net.optimize()[0]
            loss_mcts_searcher = self.mcts_searcher.optimize()
            if self.invoke_count < 1000:
                loss_tree_net = self.tree_net.optimize()[0]
                loss_mcts_searcher = self.mcts_searcher.optimize()
            if loss_tree_net > 3:
                loss_tree_net = self.tree_net.optimize()[0]
                loss_mcts_searcher = self.mcts_searcher.optimize()
            if loss_tree_net > 3:
                loss_tree_net = self.tree_net.optimize()[0]
                loss_mcts_searcher = self.mcts_searcher.optimize()
            logging.debug(f"{loss_tree_net=}, {loss_mcts_searcher=}")

    def find_base_hint(self, plan_json_pg: dict, alias: set[str], sql_vec: np.ndarray, sql: str):
        """

        @param plan_json_pg: has two key: Plan and Planning Time
        @param alias:
        @param sql_vec: &alias, from sql_encoder.encode()
        @param sql: origin sql
        @return:
        """
        # # get ids for alias, joins...
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
                                                                           id_joins_with_predicate,
                                                                           depth=leading_length)
        self.timer_record.mcts_time_list.append(self.timer.record('mcts_time_list'))

        leading_list = []
        plan_jsons = []
        leadings_utility_list = []
        # # ... 获取 hinted sql
        for leading, utility in join_list_with_predicate:
            leading_list.append(
                '/*+Leading(' + " ".join([self.sql2vec.id2aliasname[x] for x in leading[:leading_length]]) + ')*/')
            leadings_utility_list.append(utility)
            plan_jsons.append(self.db.get_cost_plan(leading_list[-1] + sql))
        # # 最后加入 cost - based plan
        plan_jsons.extend([plan_json_pg])
        leading_list.append('PG')
        leadings_utility_list.append(0)
        # # 2. 获取不确定性预测结果
        self.timer.reset('MHPE_time_list')
        plan_times = self.predict_with_uncertainty_batch(plan_jsons=plan_jsons, sql_vec=sql_vec)
        self.timer_record.mhpe_time_list.append(self.timer.record('MHPE_time_list'))
        # # 3. 排序, 选出最好的几个结果 list[( plan_time, leading, leading_utility )]
        # #    x[0][0] = mean_item
        chosen_leading_pairs = sorted(
            zip(plan_times[:config.MAX_HINT_COUNT], leading_list, leadings_utility_list, plan_jsons),
            key=lambda x: x[0][0] + self.knn.k_neighbours_sample(x[0]))
        return chosen_leading_pairs

    def predict_with_uncertainty_batch(self, plan_jsons: list[dict], sql_vec: np.ndarray):
        """
            基本就是按照公式 直接算
        @param plan_jsons:
        @param sql_vec:
        @return:
        """
        # # to tensor
        sql_feature = self.tree_net.value_network.sql_feature(sql_vec)

        fold = torchfold.Fold(cuda=False)
        multi_list = []
        for plan_json in plan_jsons:
            try:
                tree_feature = self.tree_net.tree_builder.plan_to_feature_tree(plan_json)
                # # 多次运行, 存在不确定性
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
