import sys
import matplotlib.pyplot as plt

import numpy

sys.path.append("..")

from src.basic import PostgresDB, load_json
from src.config import INPUT_WORKLOAD_NAME

# # 只有最后几个条件不一样
TEST_SQL_LIST = [[
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'DreamWorks Home Entertainment'\nAND k.keyword IN ('murder',\n'marvel-comics',\n'based-on-novel',\n'soothsayer')\nAND t.production_year > 2009;",
    '24b', [16.155, False]], [
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'DreamWorks Animation'\nAND k.keyword IN ('murder',\n'female-nudity',\n'fight',\n'tough-guy')\nAND t.production_year > 2008;",
    '24b', [13.339, False]], [
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'DreamWorks Animation'\nAND k.keyword IN ('superhero',\n'violence',\n'fight',\n'preparation')\nAND t.production_year > 2008;",
    '24b', [13.012, False]], [
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'NBC Universal Television'\nAND k.keyword IN ('magnet',\n'female-nudity',\n'superhero',\n'tough-girl')\nAND t.production_year > 2008;",
    '24b', [14.863, False]], [
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'DreamWorks Home Entertainment'\nAND k.keyword IN ('tv-special',\n'revenge',\n'superhero',\n'kung-fu')\nAND t.production_year > 2009;",
    '24b', [16.154, False]], [
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'DreamWorks Home Entertainment'\nAND k.keyword IN ('superhero',\n'marvel-comics',\n'marvel-comics',\n'abandoned-child')\nAND t.production_year > 2009;",
    '24b', [17.976, False]], [
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'DreamWorks Home Entertainment'\nAND k.keyword IN ('web',\n'marvel-comics',\n'tv-special',\n'escape-from-prison')\nAND t.production_year > 2009;",
    '24b', [17.062, False]], [
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'DreamWorks Animation'\nAND k.keyword IN ('marvel-comics',\n'fight',\n'martial-arts',\n'martial-arts')\nAND t.production_year > 2009;",
    '24b', [15.769, False]], [
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'NBC Universal Television'\nAND k.keyword IN ('claw',\n'laser',\n'tv-special',\n'martial-artist')\nAND t.production_year > 2008;",
    '24b', [13.965, False]], [
    "SELECT MIN(chn.name) AS voiced_char_name,\n       MIN(n.name) AS voicing_actress_name,\n       MIN(t.title) AS kung_fu_panda\nFROM aka_name AS an,\n     char_name AS chn,\n     cast_info AS ci,\n     company_name AS cn,\n     info_type AS it,\n     keyword AS k,\n     movie_companies AS mc,\n     movie_info AS mi,\n     movie_keyword AS mk,\n     name AS n,\n     role_type AS rt,\n     title AS t\nWHERE ci.note IN ('(voice)',\n                  '(voice: Japanese version)',\n                  '(voice) (uncredited)',\n                  '(voice: English version)')\n  AND cn.country_code ='[us]'\n  AND it.info = 'release dates'\n  AND mi.info IS NOT NULL\n  AND (mi.info LIKE 'Japan:%201%'\n       OR mi.info LIKE 'USA:%201%')\n  AND n.gender ='f'\n  AND n.name LIKE '%An%'\n  AND rt.role ='actress'\n  AND t.id = mi.movie_id\n  AND t.id = mc.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = mk.movie_id\n  AND mc.movie_id = ci.movie_id\n  AND mc.movie_id = mi.movie_id\n  AND mc.movie_id = mk.movie_id\n  AND mi.movie_id = ci.movie_id\n  AND mi.movie_id = mk.movie_id\n  AND ci.movie_id = mk.movie_id\n  AND cn.id = mc.company_id\n  AND it.id = mi.info_type_id\n  AND n.id = ci.person_id\n  AND rt.id = ci.role_id\n  AND n.id = an.person_id\n  AND ci.person_id = an.person_id\n  AND chn.id = ci.person_role_id\n  AND k.id = mk.keyword_id\n  AND t.title LIKE 'Kung Fu Panda%'\n  AND cn.name = 'DreamWorks Animation'\nAND k.keyword IN ('violence',\n'fight',\n'violence',\n'chop-socky')\nAND t.production_year > 2009;",
    '24b', [17.248, False]]]


def plot_matrix(m, x_label: str, y_label: str, title: str):
    plt.matshow(m)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def test_sql_encoder():
    from src.encoding import SQLEncoder
    print()
    sql_encoder = SQLEncoder()
    debug_dict = dict()
    sql_encoding, alias_set = sql_encoder.encode(TEST_SQL_LIST[0][0], debug_dict=debug_dict)
    print(sql_encoding)
    print(alias_set)
    plot_matrix(debug_dict['join_matrix'], 'left', 'right', 'join matrix')
    for k, v in debug_dict.items():
        print(k)
        print(v)


def test_get_plan():
    print()

    workload = load_json(INPUT_WORKLOAD_NAME)

    def dfp(node: dict, level=0):
        children = node.get('Plans', [])
        _t = '\t' * level, f"""{node['Node Type']} cost {node['Total Cost']} with {len(children)} children"""
        for child in children:
            dfp(child, level + 1)

    for i, (sql, _, _) in enumerate(workload):
        plan = PostgresDB.default().get_cost_plan(sql)
        dfp(plan["Plan"])
        print(i, len(workload), end='\r')


def test_tree_builder():
    from src.encoding import TreeBuilder

    print()

    tree_builder = TreeBuilder()
    workload = load_json(INPUT_WORKLOAD_NAME)
    for i, (sql, _, _) in enumerate(workload):
        plan = PostgresDB.default().get_cost_plan(sql)
        res = tree_builder.plan_to_feature_tree(plan)
        print(i, [len(x) for x in res])


if __name__ == '__main__':
    # test_sql_encoder()
    # test_get_plan()
    test_tree_builder()
