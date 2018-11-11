# Tensorflow LowLevelAPI

> author : ZJun

---

### Structure

- code

    - config

        - config.py
        - config_deep_wide.py 
        - config_word2vec.py
        - config_template.py

    - utils

        - utils.py
        - similar.py

    - data_process

        - data_summary.py `-data_path -data_name -data_sep -sequence_sep`
        - data_loader.py
        - raw_data_loader_example.py

    - model_tools

        - model_blocks.py
        - model_base.py

    - model_collections

        - deep_wide_model.py  `[can use this to predict ctr/cvr)]`
        - word2vec_model.py  `[can use this for (graph) embedding]`
        - model_template.py

    - embedding_analysis

        - vecs.py
        - graph.py

    - run.py

        1. deep_wide_model
            ```
            # under config file
            1. cp config_deep_wide.py config.py
            # under data_process file
            2.  python3 -B data_summary.py -data_path ../../data/ -data_name train_data 
            # under code file
            3.  python3 -B run.py -model deep_wide -data_path ../data/
            ```
        2. word2vec_model
            ```
            # under config file
            1. cp config_word2vec.py config.py
            # under data_word2vec file
            2. sh make_words.sh
            # under data_process file
            3. python3 -B data_summary.py -data_path ../../data_word2vec/ -data_name words
            # under code file
            4. python -B run.py -model word2vec -data_path ../data_word2vec/
            ```
---

- data

    - original

        - train_data
        - test_data
        - pred_data

    - generate

        [`python -B data_summary.py -data_path ../../data/ -data_name train_data`]

        - category_summary
        - numerical_summary
        - vocabulary_*
        - vocabulary_*_reindex

        [`model.predict()`]

        - prob_result


- data_word2vec

    - original

        - train_data ['center_word','target_words']
        - make_words.sh

    - generate

        [`sh make_words.sh`]

        - words 

        [`python -B data_summary.py -data_path ../../data_word2vec/ -data_name words`]

        - vocabulary_words 
        - vocabulary_words_reindex 
        - category_summary
        - numerical_summary [useless]

        [`model.write_embedding_matrix()`]

        - vocab_embedding_matrix

---

### Todo

- sequence data load and process [done]
- add lstm layer [done]
- add sequence model
- graph (relationship represent)




