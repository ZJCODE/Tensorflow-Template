#!/bin/bash



# The following command shows all available MetaGraphDef tag-sets in the SavedModel:
saved_model_cli show --dir export/1540346547

# The following command shows all available SignatureDef keys in a MetaGraphDef:
saved_model_cli show --dir export/1540346547 --tag serve

# This is very useful when you want to know the tensor key value, dtype and shape of the input tensors
# for executing the computation graph later.
saved_model_cli show --dir export/1540346547 --tag serve --signature_def serving_default

# To show all available information in the SavedModel, use the --all option.
saved_model_cli show --dir export/1540346547 --all


saved_model_cli run --dir export/1541496189 \
  --tag_set serve --signature_def serving_default \
  --input_exprs 'col1=[1];col2=[3];col3=[5];col4=[b"a"];col5=[3];col6=[b"a#d#a"];col7=[b"f"];col8=[b"w#e"]'
