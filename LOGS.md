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

## NOTE: BING 2023/5/18 下午4:04 发现

1. KNN是比较简单的, 加了点注释
2. `torchfold`是第三个第三方库, 不知道放在源码里是为了什么, 可能是因为调试吧，作者对源码做了以下修改

```python3
isinstance(arg, (torch._C._TensorBase, Variable))
## -> 
isinstance(arg, (torch.tensor._TensorBase, Variable)) 
```