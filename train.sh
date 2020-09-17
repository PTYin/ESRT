#!/bin/bash
datasets=("Musical_Instruments")
models=("LSE" "HEM" "AEM" "ZAM")
embedding_size="embed+0"
task=$1 # ('ordinary' 'user_1' 'user_3' 'user_5' 'product_1' 'product_3' 'product_5')


if [ ! -d "/home/share/yinxiangkun/log/cold_start/${task}/" ]; then
  mkdir "/home/share/yinxiangkun/log/cold_start/${task}/"
fi
if [ ! -d "/home/share/yinxiangkun/saved/cold_start/${task}/" ]; then
  mkdir "/home/share/yinxiangkun/saved/cold_start/${task}/"
fi
if [ ! -d "/home/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/" ]; then
  mkdir "/home/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/"
fi
if [ ! -d "/home/share/yinxiangkun/saved/cold_start/${task}/${embedding_size}/" ]; then
  mkdir "/home/share/yinxiangkun/saved/cold_start/${task}/${embedding_size}/"
fi

for dataset in ${datasets[@]}; do
  if [ ! -d "/home/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/${dataset}/" ]; then
    mkdir "/home/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/${dataset}/"
  fi
  if [ ! -d "/home/share/yinxiangkun/saved/cold_start/${task}/${embedding_size}/${dataset}/" ]; then
    mkdir "/home/share/yinxiangkun/saved/cold_start/${task}/${embedding_size}/${dataset}/"
  fi
  for model in ${models[@]}; do
    if [ ! -d "/home/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/${dataset}/${model}/" ]; then
      mkdir "/home/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/${dataset}/${model}/"
    fi
    python main.py --setting_file "config/${task}/${embedding_size}/${dataset}/${model}.yaml" >> "/home/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/${dataset}/${model}/train_log.txt"
    python main.py --setting_file "config/${task}/${embedding_size}/${dataset}/${model}.yaml" --decode True >> "/home/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/${dataset}/${model}/test_log.txt"
    cp "/home/share/yinxiangkun/saved/cold_start/${task}/${model}_${embedding_size}_${dataset}/test.bias_product.ranklist" "/home/share/yinxiangkun/saved/cold_start/${task}/${embedding_size}/${dataset}/${model}.ranklist"
#    ./galago-3.16-bin/bin/galago eval --judgments= "/home/share/yinxiangkun/transformed/${dataset}/seq_min_count5/seq_query_split/test.qrels" --runs+ "/home/share/yinxiangkun/saved/${embedding_size}/${dataset}/${model}.ranklist"  --metrics+recip_rank --metrics+ndcg10 --metrics+P10 >> ./log.txt
  done
done