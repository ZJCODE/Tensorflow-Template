# Tensorflow EstimatorAPI

> author : ZJun

支持如下特征自动抽取

基本：

    1. 数值特征 
    # numerical [col]
    
    2. 分桶特征 
    # bucket [[col,boundaries]...]
    
    3. 单值特征one-hot（hash/file）
    # hash category [one-hot] [[col , hash_size]...]
    # file category [one-hot] [[col,file_path]...]
    
    4. 单值特征embedding（hash/file）
    # hash embedding [embedding] [[col , hash_size , embedding_size]...] | embedding_size == -1 means dim_auto
    # file embedding [embedding] [[col , file_path,embedding_size]...]


    
进阶：

    1. 序列特征one-hot（hash/file）
    # hash sequence category [sequence one-hot ] [[col , hash_size]...]
    # file sequence category [sequence one-hot ] [[col,file_path]...]
    
    2. 序列特征embedding（hash/file）
    # hash sequence embedding [sequence embedding ] [[col , hash_size,embedding_size]...]
    # file sequence embedding [sequence embedding ] [[col,file_path,embedding_size]...]
    
    3. 序列特征multi-hot（hash/file）
    # hash multi category [multi-hot] [[col , hash_size]...]
    # file multi category [multi-hot] [[col,file_path]...]
    
    4. 针对某一特征导入本地向量
    # local vec  [[col ,vec_dim, meta_path , vec_path]...]
    
    5. 基于字典对某一特征扩展属性
    # extend feature [[col,extend_file_path,extend_features_list]]

 

 