"""
    common tools
"""
import time
import logging
import os
import math
import json
import typing
import psycopg2
import src.config as config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class PostgresDB:
    __DEFAULT_DB = None

    def __init__(self,
                 host: str = "localhost",
                 port: int = 5432,
                 user: str = "postgres",
                 password: str = "postgres",
                 dbname: str = "postgres",
                 autocommit=True,
                 latency_file_name: typing.Optional[str] = None) -> None:
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname
        )
        self.conn.autocommit = autocommit
        self.cursor = self.conn.cursor()
        self.cursor.execute("LOAD 'pg_hint_plan';")
        self.cursor.execute(f"SET geqo_threshold  = 12;")
        self.cost_plans = dict()
        self.latency_record = dict()
        self._latency_file = self.generate_latency_pool(latency_file_name)

    def get_analyse_plan(self, sql: str, timeout=300 * 1000, with_cache=False):
        """
            getAnalysePlanJson

        @param sql:
        @param timeout:
        @param with_cache:
        @return:
        """
        if config.COST_TEST_FOR_DEBUG:
            # # I don't know why this is here
            raise
        if with_cache and sql in self.latency_record:
            return self.latency_record[sql]

        timeout += 300
        try:

            self.cursor.execute(f"SET statement_timeout = {timeout};")
            self.cursor.execute(f"EXPLAIN (COSTS, FORMAT JSON, ANALYSE) {sql};")

            rows = self.cursor.fetchall()
            plan_ob = rows[0][0][0]
            plan_ob['timeout'] = False

        except psycopg2.DatabaseError as e:
            logging.info(f"DatabaseError: {e}")
            plan_ob = {
                'Planning Time': 20,
                'Plan': {
                    'Actual Total Time': config.MAX_TIME_OUT
                },
                'timeout': True
            }
        if not plan_ob['timeout']:
            self.append_latency_record(sql, plan_ob)
        return plan_ob

    def get_cost_plan(self, sql, timeout=300 * 1000):
        """
            this function looks like get_analyse_plan.

        @param sql:
        @param timeout:
        @return:
        """
        if sql in self.cost_plans:
            return self.cost_plans[sql]

        start_time = time.time()

        self.cursor.execute(f"SET statement_timeout = {timeout};")
        self.cursor.execute("SET geqo_threshold  = 12;")
        self.cursor.execute("EXPLAIN (COSTS, FORMAT JSON) " + sql)
        rows = self.cursor.fetchall()
        plan_json = rows[0][0][0]

        plan_json['Planning Time'] = time.time() - start_time

        self.cost_plans[sql] = plan_json

        return plan_json

    def get_selectivity(self, table: str, condition: str):
        """
            get selectivity of condition, it is easy to understand

            from the perspective of the number of rows in the table (statistics data)
        @param table:
        @param condition:
        @return:
        """
        if condition in self.latency_record:
            return self.latency_record[condition]
        # # query total
        self.cursor.execute("set statement_timeout = 100000;")
        self.cursor.execute(f"EXPLAIN SELECT * FROM  {table};")
        rows = self.cursor.fetchall()[0][0]
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])
        # # query select
        self.cursor.execute(f"EXPLAIN SELECT * FROM {table} WHERE {condition};")
        rows = self.cursor.fetchall()[0][0]
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        # # set
        self.append_latency_record(condition, -math.log(select_rows / total_rows))
        # # return
        return self.latency_record[condition]

    def get_latency_record(self, sql: str, timeout=300 * 1000, with_cache=False):
        if config.COST_TEST_FOR_DEBUG:
            # # I don't know why this is here
            raise
        plan_json = self.get_analyse_plan(sql, timeout, with_cache)
        return plan_json['Plan']['Actual Total Time'], plan_json['timeout']

    def get_cost(self, sql):
        """
        :param sql: a sqlSample object
        :return: the cost of sql
        """
        plan_json = self.get_analyse_plan(sql)
        return plan_json['Plan']['Total Cost'], 0

    def get_latency(self, sql, timeout=300 * 1000):
        """
        @param sql:a sqlSample object.
        @param timeout:
        @return: the latency of sql
        """
        plan_json = self.get_analyse_plan(sql, timeout)

        return plan_json['Plan']['Actual Total Time'], plan_json['timeout']

    def generate_latency_pool(self, file_name: typing.Optional[str]):
        """
            recovery latency pool from file
            and generate(reopen) a file handler for append latency record

        @param file_name: file name
        @return:
        """
        if file_name is None:
            file_name = config.LOG_LATENCY_FILE_NAME
        if not os.path.exists(file_name):
            f = open(file_name, "w")
        else:
            with open(file_name, "r") as f:
                for line in f:
                    data: list[str] = json.loads(line)
                    if '/*+Leading' not in data[0] and data[0] not in self.latency_record:
                        self.cost_plans[data[0]] = data[1]
            # # reopen file in 'append' mode
            f = open(file_name, "a")
        return f

    def append_latency_record(self, k: str, v: float | str, flush: bool = False):
        """
            push {k, v} to self.latency_record and self._latency_file
        @param k:
        @param v:
        @param flush:
        @return:
        """
        self.latency_record[k] = v
        self._latency_file.write(json.dumps([k, v]) + "\n")
        if flush:
            self._latency_file.flush()

    def __enter__(self):
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        # # do default delete
        pass

    @staticmethod
    def default() -> 'PostgresDB':
        """

        @return:
        """
        if PostgresDB.__DEFAULT_DB is None:
            PostgresDB.__DEFAULT_DB = PostgresDB(
                host=config.PG_HOST,
                port=config.PG_PORT,
                user=config.PG_USER,
                password=config.PG_PASSWORD,
                dbname=config.PG_DATABASE
            )
        return PostgresDB.__DEFAULT_DB


def load_json(json_file_name: str) -> dict | list:
    with open(json_file_name, "r") as f:
        res = json.load(f)
    return res
