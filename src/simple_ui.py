import random
import time
import typing

import pandas as pd
import streamlit as st
import st_aggrid

st.set_page_config(page_title="查询优化-展示界面")


def get_all_query_plans(sql):
    return [
        [1, 'cost-based', 0.1, 0.2, 90],
        [2, 'rule-based(/*-leading(nt, c))', 0.2, 0.3, 80],
        [3, 'rule-based(/*+leading(c, c))', 0.3, 0.4, 70],
        [4, 'rule-based(/*+leading(nt, c))', 0.4, 0.5, 60],
        [5, 'rule-based(/*+leading(nt, c))', 0.5, 0.6, 50],
        [6, 'rule-based(/*+leading(nt, c))', 0.6, 0.7, 40],
    ]


def get_query_plan_detail(plan):
    return {
        'detail': {
            '查询计划': plan,
            '详细信息': {
                '表': ['表1', '表2'],
                '条件': ['条件1', '条件2'],
            }
        },
        'tree': """
            digraph {
                a[label="A"][shape=box];
                b[label="B"][shape=box];
                c[label="C"][shape=box];
                d[label="D"][shape=box];
                e[label="E"][shape=box];
                a -> b
                b -> c
                b -> d
                d -> e
                d -> f
                f -> q
            }
        """
    }


def print_all_query_plans(res: list[list[typing.Any]], headers: list[str]):
    df = pd.DataFrame(res, columns=headers)
    # # 不显示索引列
    options_builder = st_aggrid.GridOptionsBuilder.from_dataframe(df, index=False)
    options_builder.configure_default_column(min_column_width=10,
                                             groupable=True, value=True, enableRowGroup=True,
                                             editable=False, wrapText=True)

    options_builder.configure_column("id", width=70)
    options_builder.configure_column("查询计划", width=200)
    options_builder.configure_column("生成耗费时间(s)", width=150)
    options_builder.configure_column("预期执行时间(s)", width=150)
    options_builder.configure_column("置信度(%)", width=130)

    grid_options = options_builder.build()
    st_aggrid.AgGrid(df, grid_options, theme='balham')
    return df


def main_ui():
    st.title('查询优化展示界面')
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
    # # 创建一个按钮, 获取SQL语句, SQL语句执行, 获取结果，以列表形式呈现
    if st.button('执行'):
        # # 显示loading, 用于提示用户正在执行
        # # 清除上一次的结果
        st.text('')
        with st.spinner('正在执行中...'):
            # # 模拟执行
            # # 随机等待0-5秒
            time.sleep(random.randint(0, 3))
            # # 超时显示错误信息
            if random.randint(0, 1) == 1:
                st.error('执行超时!')
                st.stop()
        # # 执行完毕, 隐藏loading
        st.success('执行完毕!')
        # # 展示JSON数据的解析结果
        st.json({
            'SQL语句': sql,
            '解析结果': {
                '表': ['表1', '表2'],
                '条件': ['条件1', '条件2'],
            }
        })
        headers = [
            'id', '查询计划', '生成耗费时间(s)', '预期执行时间(s)', '置信度(%)'
        ]
        res = get_all_query_plans(sql)
        df = print_all_query_plans(res, headers)
        # # 放一个下拉框, 让用户选择一个查询计划
        plan = st.selectbox('请选择一个查询计划: ', zip(df['id'], df['查询计划']), index=0)

        details = get_query_plan_detail(plan)
        detail_col, tree_col = st.columns(2)
        detail_col.markdown('**查询计划详情**')
        detail_col.json(details['detail'])
        tree_col.markdown('**查询计划树**')
        tree_col.graphviz_chart(details['tree'])


if __name__ == '__main__':
    main_ui()
