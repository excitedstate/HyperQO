import random
import typing
import collections

import pandas as pd
import psqlparse
import streamlit as st
import st_aggrid

import src.config as config
from src.encoding import TreeBuilder, SQLEncoder, RangeVar, WhereCondition
from src.hints import HyperQO
from src.mcts import MCTSHinterSearch
from src.net import TreeNet
from src.tree_lstm import SPINN


def get_hyper_qo_instance():
    random.seed(113)

    tree_builder = TreeBuilder()

    value_network = SPINN(head_num=config.NET_HEAD_NUM,
                          input_size=7 + 2,
                          hidden_size=config.NET_HIDDEN_SIZE,
                          table_num=50,
                          sql_size=40 * 40 + config.MAX_COLUMN_ID).to(config.DEVICE_NAME)

    tree_net = TreeNet(tree_builder=tree_builder, value_network=value_network)
    mcts_searcher = MCTSHinterSearch()

    hint_generator = HyperQO(tree_net=tree_net,
                             sql2vec=SQLEncoder(),
                             value_extractor=tree_builder.value_extractor,
                             mcts_searcher=mcts_searcher)
    return hint_generator


DEFAULT_HINT_GENERATOR = get_hyper_qo_instance()


class DotFileGenerator:
    def __init__(self, graph_name: typing.Optional[str] = None):
        self.__nodes = dict()
        self.__edges = collections.defaultdict(set)
        self.__graph_name = "TestGraph" if graph_name is None else graph_name

    def add_node(self, v_name: str, v_data: typing.Optional[str] = None):
        if v_data is None:
            self.__nodes[v_name] = v_name
        else:
            self.__nodes[v_name] = v_data

    def add_edge(self, v1_name: str, v2_name: str):
        self.__edges[v1_name].add(v2_name)

    def to_graphviz_file(self):
        res = f"digraph {self.__graph_name}{{\n"
        for v_name, v_data in self.__nodes.items():
            if "Scan" in v_data:
                res += f"\t{v_name}[label=\"{v_data}\"][shape=box][color=red][width=3];\n"
            elif "Join" in v_data or 'Nested Loop' in v_data:
                res += f"\t{v_name}[label=\"{v_data}\"][shape=box][color=blue][width=3];\n"
            else:
                res += f"\t{v_name}[label=\"{v_data}\"][shape=box][width=3];\n"
        for v1_name, v2_names in self.__edges.items():
            for v2_name in v2_names:
                res += f"\t{v1_name} -> {v2_name}\n"
        res += "}"
        return res


def get_plan_tree(plan_ob: dict):
    assert "Plan" in plan_ob, "Key Error: Plan"
    dot_file_generator = DotFileGenerator()
    undefined_mark = "undefined"

    def __get_name(level: int, node_type: str):
        return f"""{node_type.replace(' ', '')}{level}"""

    def __inner_dfs(node: dict, level=0):
        node_type = node["Node Type"]
        startup_cost = node.get("Startup Cost", undefined_mark)
        total_cost = node.get("Total Cost", undefined_mark)
        plan_rows = node.get("Plan Rows", undefined_mark)

        dot_file_generator.add_node(
            f"""{__get_name(level, node_type)}""",
            f"""{node_type}\\nstart: {startup_cost}, tol: {total_cost}\\n row: {plan_rows}"""
        )

        for child in node.get("Plans", []):
            __inner_dfs(child, level + 1)
            dot_file_generator.add_edge(f"""{__get_name(level, node_type)}""",
                                        f"""{__get_name(level + 1, child['Node Type'])}""")

    __inner_dfs(plan_ob["Plan"])
    return dot_file_generator


def get_query_plan_detail(plan):
    return {
        'detail': plan,
        'tree': get_plan_tree(plan).to_graphviz_file()
    }


def get_all_query_plans(sql):
    (pg_plan_time, pg_latency,
     mcts_time, hinter_plan_time, mhpe_time, hinter_latency,
     actual_plans, actual_time,
     chosen_leading_pairs) = DEFAULT_HINT_GENERATOR.optimize(sql)

    plan_time_record = {
        'MCTS消耗(s)': mcts_time,
        'MHPE评估消耗(s)': mhpe_time,
        '计划生成消耗(s)': hinter_plan_time,
        '计划执行消耗(s)': hinter_latency,
        '选中的查询计划': actual_plans[0],
        '选中的查询计划执行时间(s)': actual_time[0],
        'PG计划时间': pg_plan_time,
        'PG计划执行时间': pg_latency
    }
    query_plans_table = list()
    details = list()
    for i, ((mean_t, v_t, v2_t), leading, leading_utility, plan_json) in enumerate(chosen_leading_pairs):
        query_plans_table.append([i + 1, leading, mean_t, leading_utility])
        details.append(get_query_plan_detail(plan_json))
    return query_plans_table, details, plan_time_record


def print_all_query_plans(res: list[list[typing.Any]], headers: list[str]):
    df = pd.DataFrame(res, columns=headers)
    # # 不显示索引列
    options_builder = st_aggrid.GridOptionsBuilder.from_dataframe(df, index=False)
    options_builder.configure_default_column(min_column_width=10,
                                             groupable=True, value=True, enableRowGroup=True,
                                             editable=False, wrapText=True)

    options_builder.configure_column("id", width=70)
    options_builder.configure_column("查询计划", width=250)
    options_builder.configure_column("当前不确定性度量", width=150)
    options_builder.configure_column("leading可用度", width=150)

    grid_options = options_builder.build()
    st_aggrid.AgGrid(df, grid_options, theme='balham')
    return df


def get_sql_info(sql: str):
    sql_parse_result = psqlparse.parse_dict(sql)[0]["SelectStmt"]
    # # NOTE: BING 2023/5/18 下午9:55 获得FROM子句中的表名, 保存在table_list中
    # # ... 测试用例中的表数目必须大于等于2
    table_list = [str(RangeVar(x["RangeVar"])) for x in sql_parse_result["fromClause"]]

    # # NOTE: BING 2023/5/18 下午10:02 获得WHERE子句中的谓词, 保存在comparison_list中
    comparison_list = [str(WhereCondition(x)) for x in sql_parse_result["whereClause"]["BoolExpr"]["args"]]

    # # 获取SQL语
    return {
        '涉及到的表': table_list,
        '涉及到的谓词': comparison_list
    }


def main_ui():
    st.set_page_config(page_title="查询优化-展示界面", page_icon="📊")
    st.title('查询优化展示界面')
    st.markdown('当前负载: JOB-Static')
    # # 分割线
    st.markdown('---')
    st.markdown('''
        本项目旨在利用深度学习技术, 对原生的基于规则的查询优化器进行增强, 提高查询优化器的性能。
        > 本项目需配合PostgreSQL数据库使用, 本界面仅用于展示查询优化器的查询计划生成情况, 在实际中应用时, 需要修改相关变量
        
        本项目主要包含以下几个部分:
        - SQL语句的解析和编码
        - Hint的生成
        - 不同查询计划的编码
        - 代价预测模型的训练
        - 代价预测模型的应用
        
        请在下方输入SQL语句，点击执行按钮，即可查看查询计划的生成情况。
    ''')
    # # 创建一个输入框, 支持输入SQL语句
    sql = st.text_input('SQL语句', placeholder='eg: select min(id), ... from movie,... where id>1000')
    sql = sql.replace("\n", " ")
    # # 创建一个按钮, 获取SQL语句, SQL语句执行, 获取结果，以列表形式呈现
    if st.button('执行'):
        # # 显示loading, 用于提示用户正在执行
        # # 清除上一次的结果
        st.text('')
        with st.spinner('正在执行中...'):
            # # 执行
            res, details, plan_time_record = get_all_query_plans(sql)
        # # 执行完毕, 隐藏loading
        st.success('执行完毕!')

        # # 展示JSON数据的解析结果
        sql_info_col, plan_time_col = st.columns(2)

        sql_info_col.json({
            'SQL': {'statement': sql},
            '解析结果': get_sql_info(sql)
        }, expanded=False)

        plan_time_col.json(plan_time_record, expanded=False)
        headers = [
            'id',
            '查询计划',
            '当前不确定性度量',
            'leading可用度',
        ]
        df = print_all_query_plans(res, headers)
        df_zipped = list(zip(df['id'], df['查询计划']))
        # for idx, btn in enumerate(st.columns(4)):
        #     # # 设置圆角
        #     btn.markdown(
        #         f'<button id="click-to-fetch{idx}" '
        #         f'style="border-radius: 5px; background-color: #cccccc; color: #fffff; width: 90px">'
        #         f'<a href="#{df_zipped[idx][1]}">12</a></button>',
        #         unsafe_allow_html=True)

        for idx in range(len(df_zipped)):
            st.markdown('---')
            st.markdown(f'<h3 id="{df_zipped[idx][1]}">提示语 {df_zipped[idx][1]} 生成的查询计划的详细信息</h3>',
                        unsafe_allow_html=True)
            # detail_col, tree_col = st.columns(2)
            st.markdown('**查询计划详情**')
            st.json(details[idx]['detail'], expanded=False)
            st.markdown('**查询计划树**')
            st.graphviz_chart(details[idx]['tree'])


if __name__ == '__main__':
    main_ui()
