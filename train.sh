#!/bin/bash
dataset=("musical")
models=("lse" "hem" "aem" "zam")
embedding_size="8"

if [ ! -d "/home/share/yinxiangkun/log/cold_start/${embedding_size}/" ]; then
  mkdir "/home/share/yinxiangkun/log/cold_start/${embedding_size}/"
fi
if [ ! -d "/home/share/yinxiangkun/saved/cold_start/${embedding_size}/" ]; then
  mkdir "/home/share/yinxiangkun/saved/cold_start/${embedding_size}/"
fi

for (( i = 0; i < 4; i++ )); do
  if [ ! -d "/home/share/yinxiangkun/log/cold_start/${embedding_size}/${dataset[i]}/" ]; then
    mkdir "/home/share/yinxiangkun/log/cold_start/${embedding_size}/${dataset[i]}/"
  fi
  if [ ! -d "/home/share/yinxiangkun/saved/cold_start/${embedding_size}/${dataset[i]}/" ]; then
    mkdir "/home/share/yinxiangkun/saved/cold_start/${embedding_size}/${dataset[i]}/"
  fi
  for (( j = 0; j < 4; j++ )); do
    if [ ! -d "/home/share/yinxiangkun/log/cold_start/${embedding_size}/${dataset[i]}/${models[j]}/" ]; then
      mkdir "/home/share/yinxiangkun/log/cold_start/${embedding_size}/${dataset[i]}/${models[j]}/"
    fi
    python main.py --setting_file "config_${embedding_size}/${dataset[i]}/${models[j]}.yaml" >> "/home/share/yinxiangkun/log/cold_start/${embedding_size}/${dataset[i]}/${models[j]}/train_log.txt"
    python main.py --setting_file "config_${embedding_size}/${dataset[i]}/${models[j]}.yaml" --decode True >> "/home/share/yinxiangkun/log/cold_start/${embedding_size}/${dataset[i]}/${models[j]}/test_log.txt"
    cp "/home/share/yinxiangkun/saved/cold_start/${models[j]}_${embedding_size}_${dataset[i]}/test.bias_product.ranklist" "/home/share/yinxiangkun/saved/cold_start/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"
#    ./galago-3.16-bin/bin/galago eval --judgments= "/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/test.qrels" --runs+ "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"  --metrics+recip_rank --metrics+ndcg10 --metrics+P10 >> ./log.txt
  done
done