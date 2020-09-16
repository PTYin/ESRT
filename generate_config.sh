#!/bin/bash
dataset=("musical")
models=("lse" "hem" "aem" "zam")
embedding_size="8"

# create directories
if [ ! -d config_${embedding_size} ]; then
  mkdir config_${embedding_size}
  for (( i = 0; i < 4; i++ )); do
    mkdir config_${embedding_size}/${dataset[i]}
  done
fi

# generate yaml files
for (( i = 0; i < 4; i++ )); do
  # LSE
  echo "arch:
  input_feed: \"esrt.input_feed.LSEInputFeed\"
  learning_algorithm: \"esrt.models.LSE\"
  dataset_type: \"esrt.dataset.LSEDataset\"
data:
  data_dir: \"/home/share/yinxiangkun/cold_start/${dataset[i]}/seq_min_count5/\"
  input_train_dir: \"/home/share/yinxiangkun/cold_start/${dataset[i]}/seq_min_count5/seq_query_split/\"
  model_dir: \"/home/share/yinxiangkun/saved/cold_start/lse_${embedding_size}_${dataset[i]}/\"
  logging_dir: \"/home/share/yinxiangkun/log/cold_start/lse_${embedding_size}_${dataset[i]}/\"

experiment:
  subsampling_rate: 0.0001
  max_train_epoch: 50
  steps_per_checkpoint: 200
  seconds_per_checkpoint: 3600
  decode: false
  test_mode: \"product_scores\"
  rank_cutoff: 100

hparams:
  window_size: 5
  embed_size: 100
  max_gradient_norm: 5.0
  init_learning_rate: 0.5
  L2_lambda: 0.005
  query_weight: 0.5
  negative_sample: 5
  net_struct: \"LSE\"
  similarity_func: \"bias_product\"
  batch_size: 384
" > "config_${embedding_size}/${dataset[i]}/lse.yaml"
  # HEM
  echo "arch:
  input_feed: \"esrt.input_feed.HEMInputFeed\"
  learning_algorithm: \"esrt.models.HEM\"
  dataset_type: \"esrt.dataset.HEMDataset\"
data:
  data_dir: \"/home/share/yinxiangkun/cold_start/${dataset[i]}/seq_min_count5/\" #\"../seq_tmp_data/min_count5/\"
  input_train_dir: \"/home/share/yinxiangkun/cold_start/${dataset[i]}/seq_min_count5/seq_query_split/\"
  model_dir: \"/home/share/yinxiangkun/saved/cold_start/hem_${embedding_size}_${dataset[i]}/\"
  logging_dir: \"/home/share/yinxiangkun/log/cold_start/hem_${embedding_size}_${dataset[i]}/\"

experiment:
  subsampling_rate: 0.0001
  max_train_epoch: 50
  steps_per_checkpoint: 200
  seconds_per_checkpoint: 3600
  decode: false
  test_mode: \"product_scores\"
  rank_cutoff: 100

hparams:
  window_size: 5
  embed_size: 100
  max_gradient_norm: 5.0
  init_learning_rate: 0.5
  L2_lambda: 0.005
  query_weight: 0.5
  negative_sample: 5
  net_struct: \"simplified_fs\"
  similarity_func: \"bias_product\"
  batch_size: 384
" > "config_${embedding_size}/${dataset[i]}/hem.yaml"
  # AEM
  echo "arch:
  input_feed: \"esrt.input_feed.AEMInputFeed\"
  learning_algorithm: \"esrt.models.AEM\"
  dataset_type: \"esrt.dataset.AEMDataset\"
data:
  data_dir: \"/home/share/yinxiangkun/cold_start/${dataset[i]}/seq_min_count5/\"
  input_train_dir: \"/home/share/yinxiangkun/cold_start/${dataset[i]}/seq_min_count5/seq_query_split/\"
  model_dir: \"/home/share/yinxiangkun/saved/cold_start/aem_${embedding_size}_${dataset[i]}/\"
  logging_dir: \"/home/share/yinxiangkun/log/cold_start/aem_${embedding_size}_${dataset[i]}/\"

experiment:
  subsampling_rate: 0.0001
  max_train_epoch: 50
  steps_per_checkpoint: 200
  seconds_per_checkpoint: 3600
  decode: false
  test_mode: \"product_scores\"
  rank_cutoff: 100

hparams:
  window_size: 5
  embed_size: 300
  max_gradient_norm: 5.0
  init_learning_rate: 0.5
  L2_lambda: 0.005
  query_weight: 0.5
  negative_sample: 5
  net_struct: \"simplified_fs\"
  similarity_func: \"bias_product\"
  user_struct: \"asin_attention\"
  num_heads: 5
  attention_func: 'default'
  max_history_length: 10
  batch_size: 384
" > "config_${embedding_size}/${dataset[i]}/aem.yaml"
  # ZAM
  echo "arch:
  input_feed: \"esrt.input_feed.ZAMInputFeed\"
  learning_algorithm: \"esrt.models.ZAM\"
  dataset_type: \"esrt.dataset.ZAMDataset\"
data:
  data_dir: \"/home/share/yinxiangkun/cold_start/${dataset[i]}/seq_min_count5/\"
  input_train_dir: \"/home/share/yinxiangkun/cold_start/${dataset[i]}/seq_min_count5/seq_query_split/\"
  model_dir: \"/home/share/yinxiangkun/saved/cold_start/zam_${embedding_size}_${dataset[i]}/\"
  logging_dir: \"/home/share/yinxiangkun/log/cold_start/zam_${embedding_size}_${dataset[i]}/\"

experiment:
  subsampling_rate: 0.0001
  max_train_epoch: 50
  steps_per_checkpoint: 200
  seconds_per_checkpoint: 3600
  decode: false
  test_mode: \"product_scores\"
  rank_cutoff: 100

hparams:
  window_size: 5
  embed_size: 300
  max_gradient_norm: 5.0
  init_learning_rate: 0.5
  L2_lambda: 0.005
  query_weight: 0.5
  negative_sample: 5
  net_struct: \"simplified_fs\"
  similarity_func: \"bias_product\"
  user_struct: \"asin_zero_attention\"
  num_heads: 5
  attention_func: 'default'
  max_history_length: 10
  batch_size: 384
" > "config_${embedding_size}/${dataset[i]}/zam.yaml"
done