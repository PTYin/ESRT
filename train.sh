#!/bin/bash
dataset=("automotive" "cellphones" "clothing" "digital_music" "electronics" "toys")
models=("lse" "hem" "aem" "zam")
embedding_size="8"

# create directories
if [ ! -d config_${embedding_size} ]; then
  mkdir config_${embedding_size}
  for (( i = 0; i < 6; i++ )); do
    mkdir config_${embedding_size}/${dataset[i]}
  done
fi

# generate yaml files
for (( i = 0; i < 6; i++ )); do
  # LSE
  echo "arch:
  input_feed: \"esrt.input_feed.LSEInputFeed\"
  learning_algorithm: \"esrt.models.LSE\"
  dataset_type: \"esrt.dataset.LSEDataset\"
data:
  data_dir: \"/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/\"
  input_train_dir: \"/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/\"
  model_dir: \"/home/share/yinxiangkun/saved/lse_${embedding_size}_${dataset[i]}/\"
  logging_dir: \"/home/share/yinxiangkun/log/lse_${embedding_size}_${dataset[i]}/\"

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
  embed_size: ${embedding_size}
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
  data_dir: \"/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/\" #\"../seq_tmp_data/min_count5/\"
  input_train_dir: \"/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/\"
  model_dir: \"/home/share/yinxiangkun/saved/hem_${embedding_size}_${dataset[i]}/\"
  logging_dir: \"/home/share/yinxiangkun/log/hem_${embedding_size}_${dataset[i]}/\"

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
  embed_size: ${embedding_size}
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
  data_dir: \"/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/\"
  input_train_dir: \"/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/\"
  model_dir: \"/home/share/yinxiangkun/saved/aem_${embedding_size}_${dataset[i]}/\"
  logging_dir: \"/home/share/yinxiangkun/log/aem_${embedding_size}_${dataset[i]}/\"

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
  embed_size: ${embedding_size}
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
  data_dir: \"/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/\"
  input_train_dir: \"/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/\"
  model_dir: \"/home/share/yinxiangkun/saved/zam_${embedding_size}_${dataset[i]}/\"
  logging_dir: \"/home/share/yinxiangkun/log/zam_${embedding_size}_${dataset[i]}/\"

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
  embed_size: ${embedding_size}
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

if [ ! -d "/home/share/yinxiangkun/log/${embedding_size}/" ]; then
  mkdir "/home/share/yinxiangkun/log/${embedding_size}/"
fi
if [ ! -d "/home/share/yinxiangkun/saved/${embedding_size}/" ]; then
  mkdir "/home/share/yinxiangkun/saved/${embedding_size}/"
fi

for (( i = 0; i < 6; i++ )); do
  if [ ! -d "/home/share/yinxiangkun/log/${embedding_size}/${dataset[i]}/" ]; then
    mkdir "/home/share/yinxiangkun/log/${embedding_size}/${dataset[i]}/"
  fi
  if [ ! -d "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/" ]; then
    mkdir "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/"
  fi
  for (( j = 0; j < 4; j++ )); do
    if [ ! -d "/home/share/yinxiangkun/log/${embedding_size}/${dataset[i]}/${models[j]}/" ]; then
      mkdir "/home/share/yinxiangkun/log/${embedding_size}/${dataset[i]}/${models[j]}/"
    fi
    python main.py --setting_file "config_${embedding_size}/${dataset[i]}/${models[j]}.yaml" >> "/home/share/yinxiangkun/log/${embedding_size}/${dataset[i]}/${models[j]}/train_log.txt"
    python main.py --setting_file "config_${embedding_size}/${dataset[i]}/${models[j]}.yaml" --decode True >> "/home/share/yinxiangkun/log/${embedding_size}/${dataset[i]}/${models[j]}/test_log.txt"
    cp "/home/share/yinxiangkun/saved/${models[j]}_${embedding_size}_${dataset[i]}/test.bias_product.ranklist" "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"
#    ./galago-3.16-bin/bin/galago eval --judgments= "/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/test.qrels" --runs+ "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"  --metrics+recip_rank --metrics+ndcg10 --metrics+P10 >> ./log.txt
  done
done