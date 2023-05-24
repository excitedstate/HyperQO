# HyperQO

## Reference

> [1] Postgresql学习04-pg_hint_plan安装及使用、Sql优化小知识 https://blog.csdn.net/qq_45111959/article/details/125850787
>
> [2] Cost-based or Learning-based? A Hybrid Query Optimizer for Query Plan
> Selection http://dbgroup.cs.tsinghua.edu.cn/ligl/papers/optimizer-vldb23.pdf

## 项目介绍

> 详细过程请阅读论文[2]

本项目旨在利用深度学习技术, 对原生的基于规则的查询优化器进行增强, 提高查询优化器的性能。
本项目主要包含以下几个部分:

- SQL语句的解析和编码
- 候选查询计划的生成和编码
- 最优计划选择
- 代价预测模型的增量训练

### 基本思想

![img_0.png](data/pic/基本思想.png)

1. 对于复杂的查询，枚举全部的物理查询计划肯定是不现实的。
2. 采取一定算法获取“可能是最优查询计划”的几个候选查询计划；
3. 用深度神经网络构建的代价预测模型计划代价
4. 选出代价最小的查询计划，并和DBMS原生的代价做比较
5. 选择总体上代价最小的查询计划并执行

### 基本过程

![Overview.png](data/pic/Overview.png)

![Overview2.png](data/pic/Overview2.png)

### 候选查询计划: MCTS

![mcts.png](data/pic/mcts.png)

### 最优计划选择

![select_optimal.png](data/pic/select_optimal.png)

## 项目配置

## Install the PostgreSQL and pg_hint_plan

> 本节 1 - 2 引自[HyperQO](https://github.com/yxfish13/HyperQO)的README.md
>
> 本节 4 改编自[JOB](https://github.com/gregrahn/join-order-benchmark)的README.md


They made some fixes to pg_hint_plan to better support the leading hint of prefixes. The PostgreSQL and pg_hint_plan is
here[https://github.com/yxfish13/PostgreSQL12.1_hint].

### 1. Install PostgreSQL

> Tip: `bison` and `flex` is required, you should install them before compiling

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

### 3. Run PostgreSQL

```sh
# # make sure that the PG_DATA in your ENV, and pg_ctl in your PATH
pg_ctl start 
```

### 4. LOAD IMDB DATASET

#### download *gz files (unpacking not necessary)

```sh
wget ftp://ftp.fu-berlin.de/misc/movies/database/frozendata/*gz
```

#### download and unpack imdbpy and the imdbpy2sql.py script

```sh
wget https://bitbucket.org/alberanid/imdbpy/get/5.0.zip
# # if failed(especially in China), try this (i suggest you use a virtual environment):
pip install cinemagoer
# # then you have imdbpy2sql.py script, but the orm framework(i.e. SQLAlchemy) could not work well,
# # because it is too new
# # try to reinstall version 1.4.48
pip install SQLAlchemy==1.4.48
```

#### create PostgreSQL database (e.g., name imdbload):

```sh
createdb imdbload
```

#### clone the JOB repository and run `schema.sql`

```sh
git clone https://github.com/gregrahn/join-order-benchmark/
cd join-order-benchmark
psql -d imdbload -f schema.sql  # # optional
psql -d imdbload -f fkindexes.sql  # # optional
```

#### transform *gz files to relational schema (takes a while)

`PATH_TO_GZ_FILES`: the path you save gzip files

`postgres://username:password@hostname/imdbload`: the database you created

postgres协议可能不被支持，如果报错, 可以改成postgresql

```sh
imdbpy2sql.py -d PATH_TO_GZ_FILES -u postgres://username:password@hostname/imdbload
```

这个导入过程会很慢(我的电脑用了2个小时), 建议找点别的事情做

### 5. install python and packages

由于项目中用了`|`类型注解, 推荐使用`Python 3.10`, 如果你用别的版本, 可以考虑以下方案:

1. 从`__future__`导入`annotations`, 不一定能成
2. 将`float|int`改成`typing.Union[float, int]`
3. 如果1和2都不行, 建议升级Python版本, 如果您用conda, 可以使用 `conda create -n py310 python=3.10`

#### Requirements

装最新版, 除了`psqlparse`需要装`1.0rc5`版本.

`torchfold`存在兼容性问题, 需要修改部分源代码(和`torch._C._TensorBase`有关)

- `pytorch`
- `psqlparse==1.0rc5`
- `torchfold`
- `pytest`
- `psycopg2`

### 6. the last step, replace some variables in `src/config.py`

```python
# # in src/config.py, replace these variables with yours
PG_HOST = "127.0.0.1"
PG_PORT = 5432
PG_USER = 'bing'
PG_PASSWORD = "bing"
PG_DATABASE = 'imdbload'
...

PROJECT_ROOT = "/path/to/your/copy"
```

> 因为我没有GPU, 所以我没有在GPU上测试, 如果您有GPU, 可以在`src/config.py`中修改`NET_USE_GPU`为`True`
>
> 这可能会引发一些错误, 您可以尝试自己修复

## 运行

### 1. 运行Benchmark测试

```sh
python3 main.py
```

### 2. 启动图形界面

```sh
python3 start_ui.py
```

![图形界面.png](data/pic/图形界面.png)
