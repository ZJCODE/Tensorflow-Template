# Tensorflow Framework


### Note :

编辑 `conf` 目录下的文件进行数据、特征、模型配置

运行 `utils/conf.py` 查看参数配制情况

运行 `input_fn.py` 查看输入的数据情况以及自动生成的特征概况

运行 `run.py` 训练模型

模型预测代码在 `do_predict.py`

模型serving相关代码在 `serving` 目录

构建模型各种特征的基本函数在 `utils/feature.py`

构建模型结构的基本函数在 `utils/layers.py`

### Main Structure

```
Structure
|
|------ conf 
|    |--- data.yaml         [训练数据信息配置](需要编辑)
|    |--- feature.yaml      [特征自动生成配置](需要编辑)
|    |--- model.yaml        [模型训练参数配置](需要编辑)
|
|------ code
|    |--- process.py        [数据预处理,例如自动生成vocabulary文件]
|    |--- features.py       [自动生成特征的代]
|    |--- input_fn.py       [构建数据输入pipeline]
|    |--- model_fn.py       [构建模型结构,主要改写 model_fn_core 和 get_loss](需要编辑)
|    |--- run.py            [训练并导出模型,需要针对性改写feature_spec](需要编辑)
|    |--- utils
|    |  |-- conf.py         [解析配置文件]
|       |-- data.py         [提供数据处理和数据统计的模块]
|       |-- feature.py      [提供抽取特征的各种函数]
|       |-- layer.py        [提供构建模型所需的各种模型结构]
|       |-- model.py        [提供基于estimator训练和使用模型需要的一些接口]
|       |-- util.py         [提供一些常用函数]
|     
|------ data
|    |--- train             [存放训练数据,支持多个文件]
|    |--- eval              [存放验证数据,支持多个文件]
|    |--- map               [存放扩展特征使用的文件]
|    |--- vocabulary        [存放需要embedding的特征的vocabulary,一般通过process.py自动生成]
|    |--- vector            [需要将本地向量导入模型，可将meta和vec文件放入该目录]
|    |--- weight            [模型训练后，从模型中获取的权重矩阵可以存放在该目录]
|  
|------ model
|    |--- model_check       [模型的checkpoint文件]
|    |--- export            [模型的pb文件]
|
|------ serving
|    |--- README.md         [模型serving相关的简介]
|    |--- serving_cli.sh    [cli工具的使用,用于通常用来检查输出的pb文件]
|    |--- java_client       [模型serving时的java client代码示例]
|
|------ README.md           [项目简介]
```

### More About feature.yaml 

按如下方式填写配置文件，便可自动生成特征

```yaml
col_name:
  type: xxx
  transform: xxx
  parameter: xxx
```

支持如下方式的特征自动生成

```
type : continuous
|
|---transform : bucket
|            |--parameter : boundaries
|---transform : original
|            |--parameter :

type : category
|
|---transform :file_one_hot
|            |--parameter : file_path
|---transform :hash_ont_hot
|            |--parameter : hash_size
|---transform : file_embedding
|            |--parameter : file_path
|            |--parameter : embedding_size (-1 means auto)
|---transform : hash_embedding
|            |--parameter : hash_size
|            |--parameter : embedding_size (-1 means auto)
|---transform : local_vec
|            |--parameter : vec_dim
|            |--parameter : meta_path
|            |--parameter : vec_path
|---transform : extend_feature
|            |--parameter : file_path
|            |--parameter : cols_name
|            |--parameter : extend_data_sep

type : sequence
|
|---transform : file_one_hot
|            |--parameter : file_path
|---transform : hash_ont_hot
|            |--parameter : hash_size
|---transform : file_embedding
|            |--parameter : file_path
|            |--parameter : embedding_size (-1 means auto)
|---transform : hash_embedding
|            |--parameter : hash_size
|            |--parameter : embedding_size (-1 means auto)
|---transform : file_multi_hot
|            |--parameter : file_path
|---transform : hash_multi_hot
|            |--parameter : hash_size
|---transform : local_vec
|            |--parameter : vec_dim
|            |--parameter : meta_path
|            |--parameter : vec_path
|---transform : extend_feature
|            |--parameter : file_path
|            |--parameter : cols_name
|            |--parameter : extend_data_sep

```

以下给出几个例子


```yaml
col1:
  type: continuous
  transform: original
  parameter:

```
```yaml
col2:
  type: continuous
  transform: bucket
  parameter:
    boundaries: [0,5,10]
```

```yaml
col4:
  type: category
  transform: file_embedding
  parameter:
    file_path: '../data/vocabulary/vocabulary_col4'
    embedding_size: 4
```

```yaml
col7:
  type: category
  transform: local_vec
  parameter:
    meta_path: '../data/vector/meta_sample'
    vec_path: '../data/vector/vec_sample'
```

```yaml
col8:
  type: sequence
  transform: file_embedding
  parameter:
    file_path: '../data/vocabulary/vocabulary_col8'
    embedding_size: 4
```

其中`extend_feature`模块比较特殊，需要按照如下格式填写

```yaml
col_name_[extend]:
  type: xxx
  transform: xxx
  parameter:
    file_path: xxx
    cols_name: xxx
    extend_data_sep: xxx
```

例如当我们写了

```yaml
col7_[extend]:
  type: category
  transform: extend_feature
  parameter:
    file_path: '../data/map/extend_data_sample'
    cols_name: ['map_id', 'extend_1', 'extend_2']
    extend_data_sep: ' '
```
那么我们就自动获取了 `col7_extend_1` , `col7_extend_2` 这个两列，并可以对他提取新的特征

```yaml
col7_extend_1:
  type: category
  transform: file_embedding
  parameter:
    file_path: '../data/vocabulary/vocabulary_extend_1'
    embedding_size: -1

```

```yaml
col7_extend_2:
  type: category
  transform: file_embedding
  parameter:
    file_path: '../data/vocabulary/vocabulary_extend_2'
    embedding_size: -1
```

> 注 : 配置完成后,可以通过跑 `input_fn.py` 来查看配置的特征生成后有哪些，并且shape和name如何
