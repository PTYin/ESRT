#!/bin/bash
dataset=("Musical_Instruments")
models=("lse" "hem" "aem" "zam")
embedding_size="embed+0"
task=$1 # ('ordinary' 'user_1' 'user_3' 'user_5' 'product_1' 'product_3' 'product_5')

echo "dataset,model,hr@20,mrr,ndcg@20" > "${task}_${embedding_size}.csv"
for (( i = 0; i < 4; i++ )); do
  for (( j = 0; j < 4; j++ )); do
    echo -n "${dataset[i]},${models[j]}," >> "${task}_${embedding_size}.csv"
    python hr.py --judgments "/home/share/yinxiangkun/transformed/cold_start/${task}/${dataset[i]}/seq_min_count5/seq_query_split/test.qrels" --ranklist "/home/share/yinxiangkun/saved/cold_start/${task}/${embedding_size}/${dataset[i]}/${models[j]}.ranklist" --cutoff 20 | tr -d '\n' >> "${task}_${embedding_size}.csv"
    echo -n "," >> "${task}_${embedding_size}.csv"
    ./galago-3.16-bin/bin/galago eval --judgments= "/home/share/yinxiangkun/transformed/cold_start/${task}/${dataset[i]}/seq_min_count5/seq_query_split/test.qrels" --runs+ "/home/share/yinxiangkun/saved/cold_start/${task}/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"  --metrics+recip_rank --metrics+ndcg20 | awk 'NR==2{print $2}' | tr -d '\n' >> "${task}_${embedding_size}.csv"
    echo -n "," >> "${task}_${embedding_size}.csv"
    ./galago-3.16-bin/bin/galago eval --judgments= "/home/share/yinxiangkun/transformed/cold_start/${task}/${dataset[i]}/seq_min_count5/seq_query_split/test.qrels" --runs+ "/home/share/yinxiangkun/saved/cold_start/${task}/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"  --metrics+recip_rank --metrics+ndcg20 | awk 'NR==2{print $3}' >> "${task}_${embedding_size}.csv"
  done
done