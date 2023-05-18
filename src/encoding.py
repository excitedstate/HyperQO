"""
   在这个文件中实现以下编码
   1. SQLEncoder, 依赖Expr, Comparison, TargetTable, FromTable, 出现的异常用SQLEncoderError包装
   2. TreeBuilder, 依赖ValueExtractor, 出现的异常用TreeBuilderError包装
"""
import enum
import typing
import json

import numpy
import numpy as np
import psqlparse
import torch

import src.config as config
from src.basic import PostgresDB

JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES


class SQLEncoderError(Exception):
    def __init__(self, msg):
        self.__msg = msg


class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg


class Expr:
    def __init__(self, expr, list_kind=0):
        self.expr = expr
        self.list_kind = list_kind
        self.is_integer = False
        self.val = 0

    def is_col(self, ):
        return isinstance(self.expr, dict) and "ColumnRef" in self.expr

    def get_val(self, value_expr):
        if "A_Const" in value_expr:
            value = value_expr["A_Const"]["val"]
            if "String" in value:
                return "'" + value["String"]["str"].replace("'", "''") + "\'"
            elif "Integer" in value:
                self.is_integer = True
                self.val = value["Integer"]["ival"]
                return str(value["Integer"]["ival"])
            else:
                raise SQLEncoderError("unknown Value in Expr")
        elif "TypeCast" in value_expr:
            if len(value_expr["TypeCast"]['typeName']['TypeName']['names']) == 1:
                return value_expr["TypeCast"]['typeName']['TypeName']['names'][0]['String']['str'] + " '" + \
                    value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str'] + "'"
            elif value_expr["TypeCast"]['typeName']['TypeName']['typmods'][0]['A_Const']['val']['Integer']['ival'] == 2:
                return value_expr["TypeCast"]['typeName']['TypeName']['names'][1]['String']['str'] + " '" + \
                    value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str'] + "' month"
            else:
                return value_expr["TypeCast"]['typeName']['TypeName']['names'][1]['String']['str'] + " '" + \
                    value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str'] + "' year"
        else:
            print(value_expr.keys())
            raise SQLEncoderError("unknown Value in Expr")

    def get_alias_name(self, ):
        return self.expr["ColumnRef"]["fields"][0]["String"]["str"]

    def get_column_name(self, ):
        return self.expr["ColumnRef"]["fields"][1]["String"]["str"]

    def __str__(self, ):
        if self.is_col():
            return self.get_alias_name() + "." + self.get_column_name()
        elif isinstance(self.expr, dict) and "A_Const" in self.expr:
            return self.get_val(self.expr)
        elif isinstance(self.expr, dict) and "TypeCast" in self.expr:
            return self.get_val(self.expr)
        elif isinstance(self.expr, list):
            if self.list_kind == 6:
                return "(" + ", ".join([self.get_val(x) for x in self.expr]) + ")"
            elif self.list_kind == 10:
                return " AND ".join([self.get_val(x) for x in self.expr])
            else:
                raise SQLEncoderError("list kind error")

        else:
            raise SQLEncoderError("No Known type of Expr")


class RangeVar:
    """
        a unit from `FROM` clause, psqlparse express it as RangeVar Object
    """

    def __init__(self, from_table: dict):
        """
        {
            "relname": "aka_name",
            "inhOpt": 2,
            "relpersistence": "p",
            "alias": {
                "Alias": {
                    "aliasname": "an"
                }
            },
            "location": 128
        }
        @param from_table:
        """

        self.__rel_name = from_table["relname"]
        self.__alias_name = None
        if 'alias' in from_table:
            self.__alias_name = from_table["alias"]["Alias"]["aliasname"]

    @property
    def rel_name(self) -> str:
        return self.__rel_name

    @property
    def alias_name(self) -> str:
        return self.__alias_name

    def __str__(self, ):
        return self.rel_name if self.__alias_name is None else self.rel_name + " AS " + self.alias_name


@enum.unique
class ConditionType(enum.IntEnum):
    InvalidType = -1
    A_EXPR_TYPE = 0
    NULL_TEST_TYPE = 1
    BOOL_EXPR_TYPE = 2


class WhereCondition:
    """
        1. ILIKE匹配时则不区分字符串的大小写
    """

    def __parse_a_expr(self, comparison: dict):
        self.lexpr = Expr(comparison["A_Expr"]["lexpr"])
        self.column = str(self.lexpr)

        self.kind = comparison["A_Expr"]["kind"]

        if "A_Expr" not in comparison["A_Expr"]["rexpr"]:
            self.rexpr = Expr(comparison["A_Expr"]["rexpr"], self.kind)
        else:
            self.rexpr = WhereCondition(comparison["A_Expr"]["rexpr"])

        self.aliasname_list = []

        if self.lexpr.is_col():
            self.aliasname_list.append(self.lexpr.get_alias_name())
            self.column_list.append(self.lexpr.get_column_name())
        if self.rexpr.is_col():
            self.aliasname_list.append(self.rexpr.get_alias_name())
            self.column_list.append(self.rexpr.get_column_name())
        self.comp_kind = ConditionType.A_EXPR_TYPE

    def __parse_null_test(self, comparison: dict):
        self.lexpr = Expr(comparison["NullTest"]["arg"])
        self.column = str(self.lexpr)
        self.kind = comparison["NullTest"]["nulltesttype"]

        self.aliasname_list = []

        if self.lexpr.is_col():
            self.aliasname_list.append(self.lexpr.get_alias_name())
            self.column_list.append(self.lexpr.get_column_name())
        self.comp_kind = ConditionType.NULL_TEST_TYPE

    def __parse_bool_expr(self, comparison: dict):
        self.kind = comparison["BoolExpr"]["boolop"]
        self.comp_list = [WhereCondition(x) for x in comparison["BoolExpr"]["args"]]
        self.aliasname_list = []
        for comp in self.comp_list:
            if comp.lexpr.is_col():
                self.aliasname_list.append(comp.lexpr.get_alias_name())
                self.lexpr = comp.lexpr
                self.column = str(self.lexpr)
                self.column_list.append(comp.lexpr.get_column_name())
                break
        self.comp_kind = ConditionType.BOOL_EXPR_TYPE

    def __init__(self, comparison: dict):
        self.comparison = comparison
        self.column_list = list()
        self.lexpr = None
        self.column = None
        self.kind = None
        self.comp_kind = ConditionType.InvalidType
        if "A_Expr" in self.comparison:
            self.__parse_a_expr(self.comparison)
        elif "NullTest" in self.comparison:
            self.__parse_null_test(self.comparison)
        else:
            self.__parse_bool_expr(self.comparison)

    @staticmethod
    def is_col():
        return False

    def __str__(self, ):
        if self.comp_kind == 0:
            if self.kind == 0:
                Op = self.comparison["A_Expr"]["name"][0]["String"]["str"]
            elif self.kind == 7:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"] == "!~~":
                    Op = "not like"
                else:
                    Op = "like"
            elif self.kind == 8:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"] == "~~*":
                    Op = "ilike"
                else:
                    raise SQLEncoderError("Operation ERROR")
            elif self.kind == 6:
                Op = "IN"
            elif self.kind == 10:
                Op = "BETWEEN"
            else:
                print(json.dumps(self.comparison, sort_keys=True, indent=4))
                raise SQLEncoderError("Operation ERROR")
            return str(self.lexpr) + " " + Op + " " + str(self.rexpr)
        elif self.comp_kind == 1:
            if self.kind == 1:
                return str(self.lexpr) + " IS NOT NULL"
            else:
                return str(self.lexpr) + " IS NULL"
        else:
            res = ""
            for comp in self.comp_list:
                if res == "":
                    res += "( " + str(comp)
                else:
                    if self.kind == 1:
                        res += " OR "
                    else:
                        res += " AND "
                    res += str(comp)
            res += ")"
            return res


class SQLEncoder:
    COLUMN_ID_DISPATCHER = dict()
    PGCONN = PostgresDB.default()

    def __init__(self):
        self.aliasname2id = config.ALIAS_NAME2ID  # # constant
        self.id2aliasname = config.ID2ALIAS_NAME  # # constant
        self.__join_list_with_predicate = set()
        self.__join_list = set()

    def encoding(self, sql: str, debug_dict=None) -> (numpy.ndarray, set):
        """

        @param debug_dict:
        @param sql: sql query statement
        @return:
        """
        # # NOTE: BING 2023/5/18 下午9:53 初始化
        self.__reset()
        # # NOTE: BING 2023/5/18 下午9:54 解析SQL语句, 获得dict形式的结果
        sql_parse_result = psqlparse.parse_dict(sql)[0]["SelectStmt"]
        # # NOTE: BING 2023/5/18 下午9:55 获得FROM子句中的表名, 保存在table_list中
        # # ... 测试用例中的表数目必须大于等于2
        table_list = [RangeVar(x["RangeVar"]) for x in sql_parse_result["fromClause"]]
        if len(table_list) < 2:
            raise SQLEncoderError("FROM clause must have at least two tables")

        # # NOTE: BING 2023/5/18 下午10:02 获得WHERE子句中的谓词, 保存在comparison_list中
        comparison_list = [WhereCondition(x) for x in sql_parse_result["whereClause"]["BoolExpr"]["args"]]

        # # NOTE: BING 2023/5/18 下午10:00 创建一些结构体
        has_predicate = set()  # # 有谓词的表
        join_matrix = np.zeros((len(self.id2aliasname), len(self.id2aliasname)), dtype=np.float64)  # # JOIN对称矩阵
        column_selectivity = np.asarray([0] * config.MAX_COLUMN_ID, dtype=np.float64)  # # 选择性估计表
        alias_selectivity = np.zeros(len(self.id2aliasname), dtype=np.float64)  # # 选择性估计表

        aliasnames_set = set(map(lambda x: x.alias_name, table_list))  # # const, 表别名集合
        aliasname2table = {table.alias_name: table for table in table_list}  # # const, 表别名到表的映射

        for cond in comparison_list:
            if len(cond.aliasname_list) == 2:
                # # NOTE: BING 2023/5/18 下午10:35 我们认为这是一个连接条件
                left_table = cond.aliasname_list[0]
                right_table = cond.aliasname_list[1]

                left_table_id = self.aliasname2id[left_table]
                right_table_id = self.aliasname2id[right_table]
                if left_table_id < right_table_id:
                    self.__join_list.add((left_table, right_table))
                else:
                    self.__join_list.add((right_table, left_table))
                # # set join matrix
                join_matrix[left_table_id][right_table_id] = 1
                join_matrix[right_table_id][left_table_id] = 1

            else:
                # # NOTE: BING 2023/5/18 下午10:35 否则, 我们认为这是一个选择条件
                left_table = cond.aliasname_list[0]
                # # NOTE: BING 2023/5/18 下午10:39 获得该表在该条件下的选择性估计
                selectivity_query_res = SQLEncoder.PGCONN.get_selectivity(str(aliasname2table[left_table]),
                                                                          str(cond))
                has_predicate.add(left_table)

                # # 按先后顺序分配的列ID
                column_id = SQLEncoder.get_column_id(cond.column)
                column_selectivity[column_id] = column_selectivity[column_id] + selectivity_query_res
                # # 按表别名分配的列ID
                alias_name_id = self.aliasname2id[left_table]
                alias_selectivity[alias_name_id] = alias_selectivity[alias_name_id] + selectivity_query_res
        for left_table, right_table in self.__join_list:
            if left_table in has_predicate or right_table in has_predicate:
                self.__join_list_with_predicate.add((left_table, right_table))
        # # NOTE: BING 2023/5/18 下午10:40 返回结果
        if isinstance(debug_dict, dict):
            debug_dict["join_list"] = self.__join_list
            debug_dict["join_list_with_predicate"] = self.__join_list_with_predicate
            debug_dict["join_matrix"] = join_matrix
            debug_dict["column_selectivity"] = column_selectivity
        if config.MAX_COLUMN_ID == 40:
            return np.concatenate((join_matrix.flatten(), alias_selectivity)), aliasnames_set

        return np.concatenate((join_matrix.flatten(), column_selectivity)), aliasnames_set

    @property
    def join_list(self) -> set:
        return self.__join_list

    @property
    def join_list_with_predicate(self) -> set:
        return self.__join_list_with_predicate

    def __reset(self):
        self.__join_list_with_predicate = set()
        self.__join_list = set()

    @staticmethod
    def get_column_id(column):
        if column not in SQLEncoder.COLUMN_ID_DISPATCHER:
            SQLEncoder.COLUMN_ID_DISPATCHER[column] = len(SQLEncoder.COLUMN_ID_DISPATCHER)
        return SQLEncoder.COLUMN_ID_DISPATCHER[column]


class ValueExtractor:
    def __init__(self, offset=config.OFFSET, max_value=20):
        self.offset = offset
        self.max_value = max_value

    @staticmethod
    def encode(v):
        return int(np.log(2 + v) / np.log(config.MAX_TIME_OUT) * 200) / 200.
        # return int(np.log(self.offset + v) / np.log(config.max_time_out) * 200) / 200.

    @staticmethod
    def decode(v):
        # v=-(v*v<0)
        # return np.exp(v/2*np.log(config.max_time_out))#-self.offset
        return np.exp(v * np.log(config.MAX_TIME_OUT))  # -self.offset

    @staticmethod
    def cost_encode(v, min_cost, max_cost):
        return (v - min_cost) / (max_cost - min_cost)

    @staticmethod
    def cost_decode(v, min_cost, max_cost):
        return (max_cost - min_cost) * v + min_cost

    @staticmethod
    def latency_encode(v, min_latency, max_latency):
        return (v - min_latency) / (max_latency - min_latency)

    @staticmethod
    def latency_decode(v, min_latency, max_latency):
        return (max_latency - min_latency) * v + min_latency

    @staticmethod
    def rows_encode(v, min_cost, max_cost):
        return (v - min_cost) / (max_cost - min_cost)

    @staticmethod
    def rows_decode(v, min_cost, max_cost):
        return (max_cost - min_cost) * v + min_cost

    def get_plan_stats(self, data):
        return [self.encode(data["Total Cost"]), self.encode(data["Plan Rows"])]


class TreeBuilder:
    def __init__(self, value_extractor: typing.Optional[ValueExtractor] = None):
        self.value_extractor = ValueExtractor() if value_extractor is None else value_extractor
        self.value_extractor: ValueExtractor
        self.id2aliasname = config.ID2ALIAS_NAME
        self.aliasname2id = config.ALIAS_NAME2ID

    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            # # ? ? what the fuck
            for rel in self.__relations:
                if rel in node[name_key]:
                    return rel

            raise TreeBuilderError("Could not find relation name for bitmap index scan")

        raise TreeBuilderError("Cannot extract relation type from node")

    def __alias_name(self, node):
        if "Alias" in node:
            return np.asarray([self.aliasname2id[node["Alias"]]])

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Cond"  # if "Index Cond" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.aliasname2id:
                if rel + '.' in node[name_key]:
                    return np.asarray([-1])
                    # return np.asarray([self.aliasname2id[rel]])

        #     raise TreeBuilderError("Could not find relation name for bitmap index scan")
        print(node)
        raise TreeBuilderError("Cannot extract Alias type from node")

    def __featurize_join(self, node):
        assert self.is_join(node)
        # return [node["Node Type"],self.self.value_extractor.get_plan_stats(node),0,0]
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr, self.value_extractor.get_plan_stats(node)))
        feature = torch.tensor(feature, device=config.DEVICE_NAME, dtype=torch.float32).reshape(-1,
                                                                                                config.NET_INPUT_SIZE)
        return feature

    def __featurize_scan(self, node):
        assert self.is_scan(node)
        # return [node["Node Type"],self.self.value_extractor.get_plan_stats(node),self.__alias_name(node)]
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr, self.value_extractor.get_plan_stats(node)))
        feature = torch.tensor(feature, device=config.DEVICE_NAME, dtype=torch.float32).reshape(-1,
                                                                                                config.NET_INPUT_SIZE)
        return (feature,
                torch.tensor(self.__alias_name(node), device=config.DEVICE_NAME, dtype=torch.long))

    def plan_to_feature_tree(self, plan):

        # children = plan["Plans"] if "Plans" in plan else []
        if "Plan" in plan:
            plan = plan["Plan"]
        children = plan["Plan"] if "Plan" in plan else (plan["Plans"] if "Plans" in plan else [])
        if len(children) == 1:
            child_value = self.plan_to_feature_tree(children[0])
            if "Alias" in plan and plan["Node Type"] == 'Bitmap Heap Scan':
                alias_idx_np = np.asarray([self.aliasname2id[plan["Alias"]]])
                if isinstance(child_value[1], tuple):
                    raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))
                return child_value[0], torch.tensor(alias_idx_np, device=config.DEVICE_NAME, dtype=torch.long)
            return child_value
        # print(plan)
        if self.is_join(plan):
            assert len(children) == 2
            my_vec = self.__featurize_join(plan)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            # print('is_join',my_vec)
            return my_vec, left, right

        if self.is_scan(plan):
            assert not children
            # print(plan)
            s = self.__featurize_scan(plan)
            # print('is_scan',s)
            return s

        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))

    @staticmethod
    def is_join(node: dict):
        return node["Node Type"] in JOIN_TYPES

    @staticmethod
    def is_scan(node: dict):
        return node["Node Type"] in LEAF_TYPES
