启动tfs有两种方式`grpc`和`http`

**启动grpc**

```bash
/home/tfs/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    --port=2000 \
    --model_name=xxxx \
    --model_base_path=/home/xxxxx/zhangjun/xxxx/base_model

# 默认加载目录下数值最大的模型

# model_name 和 grpc 请求时
# Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setSignatureName("serving_default").setName(modelName).build();
# 里面的 modelName 对应

```

多个模型serving的时候可以按如下方式启动服务

```bash
/home/tfs/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
--port=2000 \
--model_config_file=/home/xxxxx/zhangjun/xxxx/model_config_file
```
model_config_file 按照如下方式配置
```
model_config_list: {
    config: {
        name: "xxxx",
        base_path: "/home/xxxxx/zhangjun/xxxx/base_model"
        model_platform: "tensorflow",
        model_version_policy: {
        latest: {
            num_versions: 1
           }
        }
    },
        config: {
        name: "xxxx_test",
        base_path: "/home/xxxxx/zhangjun/xxxx/test_model"
        model_platform: "tensorflow",
        model_version_policy: {
        latest: {
            num_versions: 1
           }
        }
    }
}
```
note:

model_version_policy目前支持三种选项：

- all: {} 表示加载所有发现的model；

- latest: { num_versions: n } 表示只加载最新的那n个model，也是默认选项；

- specific: { versions: m } 表示只加载指定versions的model，通常用来测试；

测试服务grpc服务:

grpc 的连接测试服务可以使用java接口，测试项目部分代码存在 `serving/java_client`

**启动http**

```bash
/home/tfs/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    --rest_api_port=3000 \
    --model_name=xxxx \
    --model_base_path=/home/xxxxx/zhangjun/xxxx/base_model
```
or
```bash
/home/tfs/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    --rest_api_port=3000 \
    --model_config_file=/home/xxxxx/zhangjun/xxxx/model_config_file
    
```
测试http

```bash
curl -XPOST http://localhost:3000/v1/models/xxxx:predict \
-d '{"signature_name":"serving_default","instances":[{"user_id":"24475643","item_id":"1269004",xxxxx}]}'
```