import math
import time
import typing

import numpy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import src.config as config

from copy import copy
from collections import namedtuple


class ValueNet(nn.Module):
    def __init__(self, in_dim, n_words=40, hidden_size=64):
        super(ValueNet, self).__init__()
        self.dim = in_dim
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.ReLU(True))
        self.output_layer = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size, hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size, 1))
        self.table_embeddings = nn.Embedding(n_words, hidden_size)
        self.hs = hidden_size
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=self.hs, out_channels=self.hs, kernel_size=5, padding=2),
                                 nn.ReLU(),
                                 nn.Conv1d(in_channels=self.hs, out_channels=self.hs, kernel_size=5, padding=2),
                                 nn.ReLU(),
                                 nn.Conv1d(in_channels=self.hs, out_channels=self.hs, kernel_size=5, padding=2),
                                 nn.MaxPool1d(kernel_size=config.MAX_HINT_COUNT))
        self.lstm = nn.LSTM(input_size=self.hs, hidden_size=self.hs, batch_first=True)

    def forward(self, q_e, j_o):
        """
            note:
                permute: 将tensor的维度换位。
        @param q_e:
        @param j_o:
        @return:
        """
        x = self.layer1(q_e).reshape(-1, self.hs)

        j_o_e = self.table_embeddings(j_o).reshape(-1, config.MAX_HINT_COUNT, self.hs)

        h = self.cnn(j_o_e.permute(0, 2, 1))
        ox = torch.cat((x, h.reshape(-1, self.hs)), dim=1)
        x = self.output_layer(ox)
        return x


class MCTSMemory:
    MCTS_SAMPLE = namedtuple('MCTSSample',
                             ('sql_feature', 'order_feature', 'label'))

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
            就是一个有限的存储
        @param args:
        @return:
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = MCTSMemory.MCTS_SAMPLE(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> list[MCTS_SAMPLE]:
        if len(self.memory) > batch_size:
            return random.sample(self.memory, batch_size)
        else:
            return self.memory

    def reset_memory(self, ):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class PlanState:
    TIME4TAKE_ACTION = 0
    TIME4GET_POSSIBLE_ACTIONS = 0

    def __init__(self,
                 number_of_tables: int,
                 query_encode: numpy.ndarray,
                 all_joins: typing.Iterable[int],
                 joins_with_predicate: typing.Iterable[int],
                 nodes: typing.Iterable[int]):
        # # init search state
        # # order_list records now path
        self.order_list = np.zeros(config.MAX_HINT_COUNT, dtype=np.int32)
        self.current_step = 0  # # idx for order_list
        self.number_of_tables = number_of_tables

        # # query encoding: from self.sql2vec.encode
        self.input_state = torch.tensor(query_encode, dtype=torch.float32).to(config.CPU_DEVICE_NAME)
        # self.query_encode = query_encode
        self.possible_actions = set()
        # self.nodes = nodes
        # # about joins, this information is included in input_state, these data structure will not change
        self.joins = list()
        self.join_matrix = dict()
        self.joins_with_predicate = list()
        self.init_data_structure4joins(all_joins, joins_with_predicate)

    def get_possible_actions(self):
        """
            获取当前步骤之后还可以添加的表
        @return:
        """
        startTime = time.time()

        if len(self.possible_actions) > 0 and self.current_step > 1:
            # # 判断是否已经获取过, 如果已经获取过, 就不必重复获取了
            return self.possible_actions

        possible_actions = set()
        if self.current_step == 1:

            for join in self.joins_with_predicate:
                if join[0] == self.order_list[0]:
                    possible_actions.add(join[1])
        elif self.current_step == 0:

            for join in self.joins_with_predicate:
                possible_actions.add(join[0])
        else:
            order_list_set = list(self.order_list)

            for join in self.joins:
                if join[0] in order_list_set and join[1] not in order_list_set:
                    possible_actions.add(join[1])
                elif join[1] in order_list_set and join[0] not in order_list_set:
                    possible_actions.add(join[0])

        self.possible_actions = possible_actions

        PlanState.TIME4GET_POSSIBLE_ACTIONS += time.time() - startTime
        return possible_actions

    def take_action(self, action: int):
        """
            action is an alias id
        @param action:
        @return:
        """
        start_time = time.time()

        new_state = copy(self)
        new_state.order_list = copy(self.order_list)
        new_state.possible_actions = copy(self.possible_actions)

        # # record path
        new_state.order_list[new_state.current_step] = action
        new_state.current_step = self.current_step + 1

        # # delete action from possible and regenerate possible
        new_state.possible_actions.remove(action)
        order_list = new_state.order_list

        for p in new_state.join_matrix[action]:
            if p not in order_list:
                new_state.possible_actions.add(p)

        PlanState.TIME4TAKE_ACTION += time.time() - start_time
        return new_state

    def is_terminal(self):
        return self.current_step == self.number_of_tables

    def init_data_structure4joins(self, all_joins, joins_with_predicate):
        """
            设置join_matrix, joins 和 joins_with_predicate
        @param all_joins:
        @param joins_with_predicate:
        @return:
        """
        for p in all_joins:
            self.join_matrix[p[0]] = set()
            self.join_matrix[p[1]] = set()
            if p[0] < p[1]:
                self.joins.append((p[0], p[1]))
            else:
                self.joins.append((p[1], p[0]))
        for p in joins_with_predicate:
            if p[0] < p[1]:
                self.joins_with_predicate.append((p[0], p[1]))
            else:
                self.joins_with_predicate.append((p[1], p[0]))
        for p in all_joins:
            self.join_matrix[p[0]].add(p[1])
            self.join_matrix[p[1]].add(p[0])


class TreeNode:
    def __init__(self, state: PlanState, parent: typing.Optional['TreeNode']):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.num_visits = 0
        self.total_reward = 0
        self.children: dict[typing.Any, TreeNode] = dict()


class MCTS:
    """
        monte carlo tree search

        1. selection
        2. expansion
        3. simulation
        4. backpropagation
    """

    def __init__(self, iteration_limit=1,
                 exploration_constant=1 / math.sqrt(16),
                 rollout_policy: typing.Optional[typing.Callable[[TreeNode], typing.Any]] = None):
        assert rollout_policy is not None, "rollout policy is None"
        assert iteration_limit >= 1, "Iteration limit must be greater than one"

        self.root = None

        # # parameters
        self.search_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.rollout = rollout_policy

        # # for debug
        self.nntime = 0
        self.nntime_no_feature = 0

    def search(self, initial_state: PlanState):
        self.root = TreeNode(initial_state, None)
        # # iterate
        for _ in range(self.search_limit):
            # # 1. Selection and Expand, this function will get a path(make node.is_terminal == True)
            node = self.select_node(self.root)

            start_time = time.time()

            # # 2. Simulation
            node, reward, nntime_no_feature = self.rollout(node)

            # # 3. update time information
            self.nntime += time.time() - start_time
            self.nntime_no_feature += nntime_no_feature
            # # 4. Back Propagation to update
            self.back_propagation(node, reward)

    def select_node(self, node: TreeNode):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                return self.expand(node)
        return node

    @staticmethod
    def expand(node: TreeNode):
        # # 获取可能的扩展
        actions = node.state.get_possible_actions()

        for action in actions:
            # # 找到第一个没被扩展的
            if action not in node.children:
                # # 扩展
                new_node = TreeNode(node.state.take_action(action), node)
                node.children[action] = new_node

                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node

        raise Exception("Should never reach here")

    @staticmethod
    def back_propagation(node: TreeNode, reward: float):
        # print(reward)
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent

    @staticmethod
    def get_best_child(node: TreeNode, exploration_constant):
        """
            根据公式选择最“好”的节点，之后执行回溯
        @param node:
        @param exploration_constant:
        @return:
        """
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.total_reward / child.num_visits + exploration_constant * math.sqrt(
                2 * math.log(node.num_visits) / child.num_visits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    @staticmethod
    def get_action(root, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action


class MCTSHinterSearch:
    def __init__(self, m_size=5000):
        self.memory = MCTSMemory(m_size)
        self.utility = []
        self.total_cnt = 0
        self.save_model_basic_name = config.LOG_FILE_NAME.split("/")[-1].split(".txt")[0]

        self.prediction_net = ValueNet(config.MCTS_INPUT_SIZE).to(config.CPU_DEVICE_NAME)
        # # optimizer and loss function
        self.optimizer = optim.Adam(self.prediction_net.parameters(), lr=3e-4, betas=(0.9, 0.999))
        self.loss_function = F.smooth_l1_loss
        self.init_prediction_net()

    def init_prediction_net(self):
        # # init
        for name, param in self.prediction_net.named_parameters():
            if len(param.shape) == 2:
                nn.init.xavier_normal(param)
            else:
                nn.init.uniform(param)

    def find_candidate_hints(self, number_of_tables, query_encode, all_joins,
                             joins_with_predicate, nodes, depth=2):
        self.total_cnt += 1
        self.utility = list()
        if self.total_cnt % 200 == 0:
            self.save_model()

        state = PlanState(number_of_tables, query_encode, all_joins, joins_with_predicate, nodes)

        mcts = MCTS(iteration_limit=int(len(state.get_possible_actions()) * config.SEARCH_SIZE),
                    rollout_policy=self.random_policy)

        mcts.search(initial_state=state)

        # # 收集 可用度 到 self.utility
        self.dfs_collect_utility(mcts.root, depth)

        # # 找出可用度 最高的的 config.HINT_TRY_TIMES 个 hints
        benefit_top_hints = sorted(self.utility, key=lambda x: x[1], reverse=True)
        return benefit_top_hints[:config.HINT_TRY_TIMES]

    def dfs_collect_utility(self, node: TreeNode, depth: int):
        """
            dfs to collect utility
        @param node:
        @param depth:
        @return:
        """
        if node.state.current_step == depth:
            node_value = node.total_reward / node.num_visits
            self.utility.append((node.state.order_list, self.e_flog(node_value)))
            return

        # # depth-first search
        for child in node.children.values():
            self.dfs_collect_utility(child, depth)

    def train(self, tree_feature: torch.Tensor, sql_vec: np.ndarray, target_value: int, alias_set: set[int]):
        """
            train one iteration
        @param tree_feature:
        @param sql_vec:
        @param target_value:
        @param alias_set:
        @return:
        """

        def plan_to_count(tree_feat):
            def recursive(feat):
                if isinstance(feat[1], tuple):
                    alias_list0 = recursive(feat=feat[1])
                    alias_list1 = recursive(feat=feat[2])
                    if len(alias_list1) == 1:
                        return alias_list0 + alias_list1
                    if len(alias_list0) == 1:
                        return alias_list1 + alias_list0
                    return []
                else:
                    return [feat[1].item()]

            return recursive(feat=tree_feat)

        # # 收集所有的features
        tree_alias = plan_to_count(tree_feature)

        if len(tree_alias) != len(alias_set):
            return

        if tree_alias[0] > tree_alias[1]:
            # # swap
            tree_alias[0], tree_alias[1] = tree_alias[1], tree_alias[0]

        tree_alias = tree_alias + [0] * (config.MAX_HINT_COUNT - len(tree_alias))

        sql_vector_tensor = torch.tensor(sql_vec, dtype=torch.float32).to(config.CPU_DEVICE_NAME)
        tree_alias_tensor = torch.tensor(tree_alias, dtype=torch.long).to(config.CPU_DEVICE_NAME)

        prediction_res = self.prediction_net(sql_vector_tensor, tree_alias_tensor)

        if target_value > config.MAX_TIME_OUT:
            target_value = config.MAX_TIME_OUT

        label = torch.tensor([(self.flog(target_value)) * 10], device=config.CPU_DEVICE_NAME, dtype=torch.float32)
        loss_value = self.loss(v=prediction_res, target=label, optimize=True)

        self.memory.push(sql_vector_tensor, tree_alias_tensor, label)

        return loss_value

    def optimize(self):
        samples = self.memory.sample(config.NET_BATCH_SIZE)

        sql_features = []
        order_features = []
        labels = []
        if len(samples) == 0:
            return

        for one_sample in samples:
            sql_features.append(one_sample.sql_feature)
            order_features.append(one_sample.order_feature)
            labels.append(one_sample.label)

        sql_feature = torch.stack(sql_features).to(config.CPU_DEVICE_NAME)
        order_feature = torch.stack(order_features).to(config.CPU_DEVICE_NAME)
        prediction_res = self.prediction_net(sql_feature, order_feature)

        label = torch.stack(labels, dim=0).reshape(-1, 1)
        loss_value = self.loss(v=prediction_res, target=label, optimize=True)
        return loss_value

    def loss(self, v, target, optimize=True):
        """
            get loss function output
            if not optimize mode: update the nn
        @param v:
        @param target:
        @param optimize:
        @return:
        """
        loss_value = self.loss_function(input=v, target=target)
        if not optimize:
            return loss_value.item()
        # # update nn.parameters
        self.optimizer.zero_grad()
        loss_value.backward()
        # # clamp
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-0.5 * 10, 0.5 * 10)
        # # step
        self.optimizer.step()
        return loss_value.item()

    def get_reward(self, state: PlanState) -> (float, float):
        """

        @param state:
        @return:
        """
        sql_vector_repr = state.input_state  # # sql encoding
        order_list_repr = torch.tensor(state.order_list, dtype=torch.long).to(config.CPU_DEVICE_NAME)

        start_time = time.time()

        with torch.no_grad():
            prediction_res = self.prediction_net(sql_vector_repr, order_list_repr)
        prediction = prediction_res.detach().cpu().numpy()[0][0] / 10
        return prediction, time.time() - start_time

    def random_policy(self, node: TreeNode) -> (TreeNode, float, float):
        """
            随机选择一条路径 并 评估
        @param node:
        @return:
        """
        t = 0
        while not node.is_terminal:
            # # 随机选择一条路径
            try:
                possible_actions = node.state.get_possible_actions()
                action = random.choice(list(possible_actions))
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(node.state))

            new_node = TreeNode(node.state.take_action(action), node)
            node.children[action] = new_node
            if len(node.state.get_possible_actions()) == len(node.children):
                node.is_fully_expanded = True
            node = new_node

        start_time = time.time()
        # # get reward of a terminal node
        reward, nntime = self.get_reward(node.state)

        t += time.time() - start_time

        return node, reward, t

    @staticmethod
    def flog(x: float | int):
        """
            log( (x + offset) / max_to ) / log(mcts_v)
            ------------------------------------------
              log( offset / max_to ) /  log(mcts_v)
            对 x 实施上述公式，得到的是一个 0~1 之间的数
        @param x:
        @return:
        """
        return int((math.log((x + config.OFFSET) / config.MAX_TIME_OUT) / math.log(config.MCTS_V))) / int(
            (math.log(config.OFFSET / config.MAX_TIME_OUT) / math.log(config.MCTS_V)))

    @staticmethod
    def e_flog(x: float | int):
        """
                  log(offset / max_to)
            x *  ----------------------  * log(mcts_v)
                      log(mcts_v)
        @param x:
        @return:
        """

        x = x * int((math.log(config.OFFSET / config.MAX_TIME_OUT) / math.log(config.MCTS_V))) * math.log(config.MCTS_V)

        return math.e ** x * config.MAX_TIME_OUT

    def save_model(self):
        torch.save(self.prediction_net.cpu().state_dict(), 'model/' + self.save_model_basic_name + ".pth")

    def load_model(self):
        self.prediction_net.load_state_dict(torch.load('model/' + self.save_model_basic_name + ".pth"))
