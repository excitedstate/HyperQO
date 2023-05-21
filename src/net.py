import math
import random

import numpy as np
import torch
import torch.nn as nn
import torchfold
import torch.optim as optim

from collections import namedtuple

import src.config as config
from src.encoding import TreeBuilder
from src.tree_lstm import SPINN


class MSEVAR(nn.Module):
    def __init__(self, var_weight: int):
        """
            this is a new loss function
        @param var_weight:
        """
        super(MSEVAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, multi_value, target, var):
        """
            loss = exp(-var_weight * var) * (multi_value - target) ** 2 + var_weight * var
            return loss.mean()
        @param multi_value:
        @param target:
        @param var:
        @return:
        """
        # # 1. define variables weights, now var_wei = 0
        var_wei = (self.var_weight * var).reshape(-1, 1)
        # # 2. so loss = loss1 = (multi_value - target) ** 2, that is MSE
        loss1 = torch.mul(torch.exp(-var_wei), (multi_value - target) ** 2)
        loss2 = var_wei
        loss3 = 0
        loss = (loss1 + loss2 + loss3)
        return loss.mean()


class ReplayMemory(object):
    TRANSITION = namedtuple('Transition', ('tree_feature', 'sql_feature', 'target_feature', 'mask', 'weight'))

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, *args):
        """
            save a transition, save more memory when memory is not full
        @param args:
        @return:
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = ReplayMemory.TRANSITION(*args)
        self.position = (self.position + 1) % self.capacity

    def weight_sample(self, batch_size: int):
        """
            sample batch_size elements from memory, the probability of each element is proportional to its weight
        @param batch_size:
        @return:
        """
        weight = []
        current_weight = 0
        for x in self.memory:
            current_weight += x.weight
            weight.append(current_weight)
        # # 计算每个元素的权重占比
        for idx in range(len(self.memory)):
            weight[idx] = weight[idx] / current_weight
        # # 从0~1中随机取batch_size个数, 每个数的概率为weight中对应位置的元素
        return random.choices(
            population=list(range(len(self.memory))),
            weights=weight,
            k=batch_size
        )

    def sample(self, batch_size) -> (list['ReplayMemory.TRANSITION'], list[int]):
        if len(self.memory) > batch_size:
            normal_batch = batch_size // 2
            # # 获取normal_batch个随机数, 这样取随机数的话, 会有重复的
            idx_list1 = []
            for _ in range(normal_batch):
                idx_list1.append(random.randint(0, normal_batch - 1))
            # # 按权重取batch_size - normal_batch个随机数
            idx_list2 = self.weight_sample(batch_size=batch_size - normal_batch)
            idx_list = idx_list1 + idx_list2
            # # 获取对应位置的元素, 可能会有重复的, 可能重复是被设计的
            res = []
            for idx in idx_list:
                res.append(self.memory[idx])
            return res, idx_list
        else:
            return self.memory, list(range(len(self.memory)))

    def update_weight(self, idx_list, weight_list):
        """
            将idx_list中对应位置元素的weight更新为weight_list中的元素
        @param idx_list:
        @param weight_list:
        @return:
        """
        for idx, weight in zip(idx_list, weight_list):
            # # seems to be better
            self.memory[idx] = ReplayMemory.TRANSITION(*self.memory[idx][:-1], weight)
            # self.memory[idx] = self.memory[idx]._replace(weight=weight)

    def reset_memory(self, ):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class TreeNet:
    def __init__(self, tree_builder: TreeBuilder, value_network: SPINN):
        self.tree_builder = tree_builder  # encoding.TreeBuilder
        self.value_network = value_network  # tree_lstm.SPINN
        self.memory = ReplayMemory(config.MEM_SIZE)
        # # for training
        self.optimizer = optim.Adam(value_network.parameters(), lr=3e-4, betas=(0.9, 0.999))
        self.loss_function = MSEVAR(config.VAR_WEIGHT)

    def loss(self, multi_value, target, var, optimize=True):
        """
            calculate the loss
        @param multi_value: res mask
        @param target: target mask
        @param var: multi value
        @param optimize: if optimize is True, then update the parameters, 控制是否更新参数
        @return:
        """
        # # var是原来的multi_value
        loss_value = self.loss_function(multi_value=multi_value, target=target, var=var)
        # # 判断是否更新参数(训练过程中更新参数, 预测过程中不更新参数)
        if not optimize:
            return loss_value.item()
        # # 经典的神经网络训练
        self.optimizer.zero_grad()
        loss_value.backward()
        # # 梯度裁剪 clamp_ 是 clamp 的 in-place 版本
        # # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-2, 2)
        # # 更新参数
        self.optimizer.step()
        return loss_value.item()

    def train(self, plan_json: dict, sql_vec: np.ndarray, target_value: float, mask: torch.tensor, optimize=False):
        """
            train the model(one iteration)
        @param plan_json: dict, has two key: Plan and Planning Time
        @param sql_vec: a vector repr of sql, sql_vec = sql_encoder.encode(sql), 包含join信息和谓词信息
        @param target_value: value_extractor.encode的结果, a float
        @param mask: a vector of 0 and 1, 1 means the value is valid
        @param optimize: if optimize is True, then update the parameters, 控制是否更新参数
        @return:
        """
        # # 1. get the feature of plan
        plan_tree_feature = self.tree_builder.plan_to_feature_tree(plan_json)
        # # 2. 将numpy表示的 sql repr 转换为tensor
        sql_feature = self.value_network.sql_feature(sql_vec)
        # # 3. 将plan_feature和sql_feature输入到网络中, 得到 LSTM-Tree 的编码, 多头的输出
        multi_value = self.plan_to_value(tree_feature=plan_tree_feature, sql_feature=sql_feature)
        multi_value_res = multi_value[:, :config.NET_HEAD_NUM]  # # todo 注意观察这里是多少维度的
        # # 4. 将target_value转换为target_feature, 就是复制 head_num 份 (多头)
        target_feature = self.value_network.target_vec(target_value).reshape(1, -1)

        # # 5. 计算loss, 多头的输出
        loss_value = self.loss(multi_value=multi_value_res * mask, target=target_feature * mask,
                               optimize=optimize, var=multi_value[:, config.NET_HEAD_NUM])
        # # 6. 计算mean和variance
        mean, variance = self.mean_and_variance(multi_value=multi_value_res)
        # # 7. 将数据存入memory中
        self.memory.push(plan_tree_feature, sql_feature, target_feature, mask, abs(mean - target_value))

        # # 之前是用math.e**的
        # # 8. 返回loss, mean, variance以及结果
        return loss_value, mean, variance, np.exp(multi_value[:, config.NET_HEAD_NUM].item())

    def optimize(self):
        """
            about torchfold
            1. Analogous to TensorFlow Fold, implements dynamic batching with super simple interface.
               ...Replace every direct call in your computation to nn module with f.add('function name', arguments).
               ...It will construct an optimized version of computation and on f.apply will dynamically batch and
               ...execute the computation on given nn module.

            about this

        @return:
        """
        # # 使用Fold 用于 动态批处理
        fold = torchfold.Fold(cuda=False)
        # # 取出NET_BATCH_SIZE个样本
        samples, samples_idx = self.memory.sample(config.NET_BATCH_SIZE)
        target_features = []
        masks = []

        multi_list = []
        target_values = []
        for s in samples:
            s: ReplayMemory.TRANSITION

            multi_value = self.plan_to_value_fold(tree_feature=s.tree_feature,
                                                  sql_feature=s.sql_feature, fold=fold)
            masks.append(s.mask)
            target_features.append(s.target_feature)
            target_values.append(s.target_feature.mean().item())
            multi_list.append(multi_value)
        # #
        multi_value = fold.apply(self.value_network, [multi_list])[0]
        mask = torch.cat(masks, dim=0)
        target_feature = torch.cat(target_features, dim=0)
        loss_value = self.loss(multi_value=multi_value[:, :config.NET_HEAD_NUM] * mask, target=target_feature * mask,
                               optimize=True, var=multi_value[:, config.NET_HEAD_NUM])

        mean, variance = self.mean_and_variance(multi_value=multi_value[:, :config.NET_HEAD_NUM])
        mean_list = [mean] if isinstance(mean, float) else [x.item() for x in mean]
        # # 更新memory中的权重
        new_weight = [abs(x - target_values[idx]) * target_values[idx] for idx, x in enumerate(mean_list)]
        self.memory.update_weight(samples_idx, new_weight)

        return loss_value, mean, variance, torch.exp(multi_value[:, config.NET_HEAD_NUM]).data.reshape(-1)

    @staticmethod
    def plan_to_value_fold(tree_feature, sql_feature, fold):
        """
            similar to  plan_to_value but batching
        @param tree_feature:
        @param sql_feature:
        @param fold:
        @return:
        """

        def recursive(tree_feat):
            if isinstance(tree_feat[1], tuple):
                feature = tree_feat[0]
                h_left, c_left = recursive(tree_feat=tree_feat[1]).split(2)
                h_right, c_right = recursive(tree_feat=tree_feat[2]).split(2)
                return fold.add('tree_node', h_left, c_left, h_right, c_right, feature)
            else:
                feature = tree_feat[0]
                h_left, c_left = fold.add('leaf', tree_feat[1]).split(2)
                h_right, c_right = fold.add('zero_hc', 1).split(2)
                return fold.add('tree_node', h_left, c_left, h_right, c_right, feature)

        plan_feature, c = recursive(tree_feat=tree_feature).split(2)
        # sql_feature = fold.add('sql_feature',sql_vec)
        multi_value = fold.add('logits', plan_feature, sql_feature)
        return multi_value

    @staticmethod
    def mean_and_variance(multi_value: torch.Tensor):
        """
            calculate the mean and variance of multi_value
        @param multi_value: head_num, 1
        @return:
        """
        mean_value = torch.mean(multi_value, dim=1).reshape(-1, 1)
        variance = torch.sum((multi_value - mean_value) ** 2, dim=1) / multi_value.shape[1]
        if mean_value.shape[0] == 1:
            return mean_value.item(), variance.item() ** 0.5
        else:
            return mean_value.data, variance.data ** 0.5

    def plan_to_value(self, tree_feature: tuple, sql_feature):
        """

        @param tree_feature: plan_to_feature_tree
        @param sql_feature:
        @return:
        """

        def recursive(tree_feat: tuple):
            """
                return tree-lstm, this is a recursive progress
                return two vectors: h, c
            @param tree_feat:
            @return:
            """
            if isinstance(tree_feat[1], tuple):
                feature = tree_feat[0]
                h_left, c_left = recursive(tree_feat=tree_feat[1])
                h_right, c_right = recursive(tree_feat=tree_feat[2])
                return self.value_network.tree_node(h_left, c_left, h_right, c_right, feature)
            else:
                # # leaf node
                feature = tree_feat[0]
                h_left, c_left = self.value_network.leaf(tree_feat[1])
                h_right, c_right = self.value_network.zero_hc()

                return self.value_network.tree_node(h_left, c_left, h_right, c_right, feature)

        plan_feature = recursive(tree_feat=tree_feature)
        # # transform plan_feature to vector, combine tree encoding and sql feature
        multi_value = self.value_network.logits(plan_feature[0], sql_feature)
        return multi_value
