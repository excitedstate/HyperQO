# 研读日志

## 已经处理过的文件

-[x] [ImportantConfig.py](temp/v0/ImportantConfig.py) -> [config.py](src/config.py)
-[x] [PGUtils.py](temp/v0/PGUtils.py) -> [basic.py](src/basic.py): 存在较多没有用到的函数
-[ ] [sql2fea.py](temp/v0/sql2fea.py) -> [encoding.py](src/encoding.py)
-[ ] [JOBParser.py](temp/v0/JOBParser.py) -> [encoding.py](src/encoding.py)
    -[ ] `Table`和`DB`没有用到
    -[ ] Expr被Comparison引用
    -[ ] TargetTable和 FromTable 被外部引用
    -[ ] Comparison被外部和自身调用

## UNUSED

- [x] [KNN.py](temp/v0/KNN.py)

## 主文件分析

## NOTE: BING 2023/5/17 下午4:24 复制

1. 先处理简单的, 即关于和PG连接的部分, redefined in [basic.py](src/basic.py)
2. 文件分类完成, 明天开始精读并测试

## NOTE: BING 2023/5/22 下午7:30 对程序做进一步简化, 分析性能瓶颈

1. 去掉标志：COST_TEST_FOR_DEBUG