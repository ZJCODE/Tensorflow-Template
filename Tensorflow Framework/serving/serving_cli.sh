#!/bin/bash

# 在启动服务前，我们可以先对模型文件做一些简单的查看，例如

# The following command shows all available MetaGraphDef tag-sets in the SavedModel:
saved_model_cli show --dir export/1543993136

# The following command shows all available SignatureDef keys in a MetaGraphDef:
saved_model_cli show --dir export/1543993136 --tag serve

# This is very useful when you want to know the tensor key value, dtype and shape of the input tensors
# for executing the computation graph later.
saved_model_cli show --dir export/1543993136 --tag serve --signature_def serving_default

# To show all available information in the SavedModel, use the --all option.
saved_model_cli show --dir export/1543996911 --all
saved_model_cli show --dir export_raw_input/1543996913 --all

#saved_model_cli run --dir export/1543993136 \
#  --tag_set serve --signature_def serving_default \
#  --input_exprs 'col1=[1];col2=[3];col3=[5];col4=[b"a"];col5=[3];col6=[b"a#d#a"];col7=[b"f"];col8=[b"w#e"]'

saved_model_cli run --dir export/1543993136 \
  --tag_set serve --signature_def serving_default \
  --input_examples 'examples = [{"col1":[1],"col2":[3],"col3":[5],"col4":["a"],"col5":[3],"col6":["a#d#a"],"col7":["f"],"col8":["w#e"]}]'

