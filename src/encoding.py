"""
   在这个文件中实现以下编码
   1. SQL表示的查询到向量的编码
"""
import numpy
import numpy as np
import psqlparse
import torch

import src.config as config
from src.basic import PostgresDB


class Expr:
    def __init__(self, expr, list_kind=0):
        self.expr = expr
        self.list_kind = list_kind
        self.isInt = False
        self.val = 0

    def is_col(self, ):
        return isinstance(self.expr, dict) and "ColumnRef" in self.expr

    def get_val(self, value_expr):
        if "A_Const" in value_expr:
            value = value_expr["A_Const"]["val"]
            if "String" in value:
                return "'" + value["String"]["str"].replace("'", "''") + "\'"
            elif "Integer" in value:
                self.isInt = True
                self.val = value["Integer"]["ival"]
                return str(value["Integer"]["ival"])
            else:
                raise "unknown Value in Expr"
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
            raise "unknown Value in Expr"

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
                return "(" + ",\n".join([self.get_val(x) for x in self.expr]) + ")"
            elif self.list_kind == 10:
                return " AND ".join([self.get_val(x) for x in self.expr])
            else:
                raise "list kind error"

        else:
            raise "No Known type of Expr"


class TargetTable:
    def __init__(self, target):
        """
        {
            'location': 7,
            'name': 'alternative_name',
            'val': {'FuncCall': {
                    'funcname': [{
                            'String': {'str': 'min'}
                        }],'args': [{
                        'ColumnRef': {
                            'fields': [
                                {
                                    'String': {'str': 'an'}
                                }, {
                                    'String': {'str': 'name'}
                                }
                            ],
                        'location': 11
                        }
                    }],
                    'location': 7
                }
            }
        }
        """
        self.target = target

    #         print(self.target)

    def get_val(self, ):
        columnRef = self.target["val"]["FuncCall"]["args"][0]["ColumnRef"]["fields"]
        return columnRef[0]["String"]["str"] + "." + columnRef[1]["String"]["str"]

    def __str__(self, ):
        try:
            return self.target["val"]["FuncCall"]["funcname"][0]["String"][
                "str"] + "(" + self.get_val() + ")" + " AS " + self.target['name']
        except Exception as e:
            print(e)
            if "FuncCall" in self.target["val"]:
                return "count(*)"
            else:
                return "*"


class FromTable:
    def __init__(self, from_table):
        """
            {'alias':
                {'Alias':
                 {'aliasname': 'an'}},
                 'location': 168,
                  'inhOpt': 2,
                   'relpersistence': 'p',
                    'relname': 'aka_name'
            }
        """
        self.from_table = from_table
        if 'alias' not in self.from_table:
            self.from_table['alias'] = {'Alias': {'aliasname': from_table['relname']}}

    def get_full_name(self, ):
        return self.from_table["relname"]

    def get_alias_name(self, ):
        return self.from_table["alias"]["Alias"]["aliasname"]

    def __str__(self, ):
        try:
            return self.get_full_name() + " AS " + self.get_alias_name()
        except Exception as e:
            print(self.from_table)
            raise e


class Comparison:
    def __init__(self, comparison):
        self.comparison = comparison
        self.column_list = []
        if "A_Expr" in self.comparison:
            self.lexpr = Expr(comparison["A_Expr"]["lexpr"])
            self.column = str(self.lexpr)
            self.kind = comparison["A_Expr"]["kind"]
            if "A_Expr" not in comparison["A_Expr"]["rexpr"]:
                self.rexpr = Expr(comparison["A_Expr"]["rexpr"], self.kind)
            else:
                self.rexpr = Comparison(comparison["A_Expr"]["rexpr"])

            self.aliasname_list = []

            if self.lexpr.is_col():
                self.aliasname_list.append(self.lexpr.get_alias_name())
                self.column_list.append(self.lexpr.get_column_name())

            if self.rexpr.is_col():
                self.aliasname_list.append(self.rexpr.get_alias_name())
                self.column_list.append(self.rexpr.get_column_name())

            self.comp_kind = 0
        elif "NullTest" in self.comparison:
            self.lexpr = Expr(comparison["NullTest"]["arg"])
            self.column = str(self.lexpr)
            self.kind = comparison["NullTest"]["nulltesttype"]

            self.aliasname_list = []

            if self.lexpr.is_col():
                self.aliasname_list.append(self.lexpr.get_alias_name())
                self.column_list.append(self.lexpr.get_column_name())
            self.comp_kind = 1
        else:
            #             "boolop"
            self.kind = comparison["BoolExpr"]["boolop"]
            self.comp_list = [Comparison(x)
                              for x in comparison["BoolExpr"]["args"]]
            self.aliasname_list = []
            for comp in self.comp_list:
                if comp.lexpr.is_col():
                    self.aliasname_list.append(comp.lexpr.get_alias_name())
                    self.lexpr = comp.lexpr
                    self.column = str(self.lexpr)
                    self.column_list.append(comp.lexpr.get_column_name())
                    break
            self.comp_kind = 2

    def is_col(self, ):
        return False

    def __str__(self, ):
        if self.comp_kind == 0:
            Op = ""
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
                    raise
            elif self.kind == 6:
                Op = "IN"
            elif self.kind == 10:
                Op = "BETWEEN"
            else:
                import json
                print(json.dumps(self.comparison, sort_keys=True, indent=4))
                raise "Operation ERROR"
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
        pass

    def get_column_id(self, column):
        if column not in self.COLUMN_ID_DISPATCHER:
            self.COLUMN_ID_DISPATCHER[column] = len(self.COLUMN_ID_DISPATCHER)
        return self.COLUMN_ID_DISPATCHER[column]

    def encoding(self, sql: str) -> numpy.array:
        # startTime = time.time()
        parse_result = psqlparse.parse_dict(sql)[0]["SelectStmt"]
        target_table_list = [TargetTable(x["ResTarget"]) for x in parse_result["targetList"]]
        from_table_list = [FromTable(x["RangeVar"]) for x in parse_result["fromClause"]]
        if len(from_table_list) < 2:
            return
        aliasname2fullname = {}

        id2aliasname = config.id2aliasname
        aliasname2id = config.aliasname2id

        join_list = set()
        aliasnames_root_set = set([x.get_alias_name() for x in from_table_list])

        alias_selectivity = np.asarray([0] * len(id2aliasname), dtype=np.float)
        aliasname2fromtable = {}
        for table in from_table_list:
            aliasname2fromtable[table.get_alias_name()] = table
            aliasname2fullname[table.get_alias_name()] = table.get_full_name()

        comparison_list = [Comparison(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]
        join_matrix = np.zeros((len(id2aliasname), len(id2aliasname)), dtype=np.float)
        count_selectivity = np.asarray([0] * config.max_column, dtype=np.float)
        has_predicate = set()
        join_list_with_predicate = set()
        for comparison in comparison_list:
            if len(comparison.aliasname_list) == 2:
                left_aliasname = comparison.aliasname_list[0]
                right_aliasname = comparison.aliasname_list[1]
                idx0 = aliasname2id[left_aliasname]
                idx1 = aliasname2id[right_aliasname]
                if idx0 < idx1:
                    join_list.add((left_aliasname, right_aliasname))
                else:
                    join_list.add((right_aliasname, left_aliasname))
                join_matrix[idx0][idx1] = 1
                join_matrix[idx1][idx0] = 1
            else:
                left_aliasname = comparison.aliasname_list[0]
                alias_selectivity[aliasname2id[left_aliasname]] = alias_selectivity[aliasname2id[
                    left_aliasname]] + self.PGCONN.getSelectivity(
                    str(aliasname2fromtable[comparison.aliasname_list[0]]), str(comparison))
                has_predicate.add(left_aliasname)
                count_selectivity[self.get_column_id(comparison.column)] = count_selectivity[
                                                                               self.get_column_id(
                                                                                   comparison.column)] + \
                                                                           self.PGCONN.getSelectivity(
                                                                               str(aliasname2fromtable[
                                                                                       comparison.aliasname_list[
                                                                                           0]]), str(comparison))
        for arr_join in join_list:
            if arr_join[0] in has_predicate or arr_join[1] in has_predicate:
                join_list_with_predicate.add(arr_join)
        if config.max_column == 40:
            return np.concatenate((join_matrix.flatten(), alias_selectivity)), aliasnames_root_set
        # print(np.concatenate((join_matrix.flatten(),count_selectivity)).shape)
        return np.concatenate((join_matrix.flatten(), count_selectivity)), aliasnames_root_set



JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES


class ValueExtractor:
    def __init__(self, offset=config.offset, max_value=20):
        self.offset = offset
        self.max_value = max_value

    # def encode(self,v):
    #     return np.log(self.offset+v)/np.log(2)/self.max_value
    # def decode(self,v):
    #     # v=-(v*v<0)
    #     return np.exp(v*self.max_value*np.log(2))#-self.offset
    def encode(self, v):
        return int(np.log(2 + v) / np.log(config.max_time_out) * 200) / 200.
        return int(np.log(self.offset + v) / np.log(config.max_time_out) * 200) / 200.

    def decode(self, v):
        # v=-(v*v<0)
        # return np.exp(v/2*np.log(config.max_time_out))#-self.offset
        return np.exp(v * np.log(config.max_time_out))  # -self.offset

    def cost_encode(self, v, min_cost, max_cost):
        return (v - min_cost) / (max_cost - min_cost)

    def cost_decode(self, v, min_cost, max_cost):
        return (max_cost - min_cost) * v + min_cost

    def latency_encode(self, v, min_latency, max_latency):
        return (v - min_latency) / (max_latency - min_latency)

    def latency_decode(self, v, min_latency, max_latency):
        return (max_latency - min_latency) * v + min_latency

    def rows_encode(self, v, min_cost, max_cost):
        return (v - min_cost) / (max_cost - min_cost)

    def rows_decode(self, v, min_cost, max_cost):
        return (max_cost - min_cost) * v + min_cost


value_extractor = ValueExtractor()


def get_plan_stats(data):
    return [value_extractor.encode(data["Total Cost"]), value_extractor.encode(data["Plan Rows"])]


class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg


def is_join(node):
    return node["Node Type"] in JOIN_TYPES


def is_scan(node):
    return node["Node Type"] in LEAF_TYPES


# fasttext
class PredicateEncode:
    def __init__(self, ):
        pass

    def stringEncoder(self, string_predicate):
        return torch.tensor([0, 1] + [0] * config.hidden_size, device=config.device).float()
        pass

    def floatEncoder(self, float1, float2):
        return torch.tensor([float1, float2] + [0] * config.hidden_size, device=config.device).float()
        pass


class TreeBuilder:
    def __init__(self):
        self.__stats = get_plan_stats
        self.id2aliasname = config.id2aliasname
        self.aliasname2id = config.aliasname2id

    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
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
                    return np.asarray([self.aliasname2id[rel]])

        #     raise TreeBuilderError("Could not find relation name for bitmap index scan")
        print(node)
        raise TreeBuilderError("Cannot extract Alias type from node")

    def __featurize_join(self, node):
        assert is_join(node)
        # return [node["Node Type"],self.__stats(node),0,0]
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr, self.__stats(node)))
        feature = torch.tensor(feature, device=config.device, dtype=torch.float32).reshape(-1, config.input_size)
        return feature

    def __featurize_scan(self, node):
        assert is_scan(node)
        # return [node["Node Type"],self.__stats(node),self.__alias_name(node)]
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr, self.__stats(node)))
        feature = torch.tensor(feature, device=config.device, dtype=torch.float32).reshape(-1, config.input_size)
        return (feature,
                torch.tensor(self.__alias_name(node), device=config.device, dtype=torch.long))

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
                return (child_value[0], torch.tensor(alias_idx_np, device=config.device, dtype=torch.long))
            return child_value
        # print(plan)
        if is_join(plan):
            assert len(children) == 2
            my_vec = self.__featurize_join(plan)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            # print('is_join',my_vec)
            return (my_vec, left, right)

        if is_scan(plan):
            assert not children
            # print(plan)
            s = self.__featurize_scan(plan)
            # print('is_scan',s)
            return s

        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))
