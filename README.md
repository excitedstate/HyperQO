# HyperQO

## Reference

> [1] Postgresql学习04-pg_hint_plan安装及使用、Sql优化小知识 https://blog.csdn.net/qq_45111959/article/details/125850787

# Requirements

- Pytorch 1.0
- Python 3.7
- torchfold
- psqlparse

## Install the PostgreSQL and pg_hint_plan

We made some fixes to pg_hint_plan to better support the leading hint of prefixes. The PostgreSQL and pg_hint_plan is
here[https://github.com/yxfish13/PostgreSQL12.1_hint].

### 1. Install PostgreSQL

> tip: `bison` and `flex` is required, you should install them before compiling

 ```sh
 cd postgresql-12.1/
 ./configure --prefix=/usr/local/pgsql --with-segsize=16 --with-blocksize=32 --with-wal-segsize=64 --with-wal-blocksize=64 --with-libedit-preferred  --with-python --with-openssl --with-libxml --with-libxslt --enable-thread-safety --enable-nls=en_US.UTF-8
 make
 make install
 ```

### 2. Install pg_hint_plan

 ```sh
 cd postgresql-12.1/pg_hint_plan-REL12_1_3_6/
 ./configure
 make install
 ```

## Running

1. configurate the ImportantConfig.py
2. run

 ```sh
     python3 run_mcts.py
 ```

