#!/bin/bash
dataset=("automotive" "cellphones" "clothing" "digital_music" "electronics" "toys")
models=("lse" "hem" "aem" "zam")
embedding_size="embed+0"

echo "dataset,model,hr@20,mrr,ndcg@20" > "${embedding_size}.csv"
for (( i = 0; i < 6; i++ )); do
  for (( j = 0; j < 4; j++ )); do
    echo -n "${dataset[i]},${models[j]}," >> "${embedding_size}.csv"
    python hr.py --judgments "/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/test.qrels" --ranklist "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/${models[j]}.ranklist" --cutoff 20 | tr -d '\n' >> "${embedding_size}.csv"
    echo -n "," >> "${embedding_size}.csv"
    ./galago-3.16-bin/bin/galago eval --judgments= "/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/test.qrels" --runs+ "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"  --metrics+recip_rank --metrics+ndcg20 | awk 'NR==2{print $2}' | tr -d '\n' >> "${embedding_size}.csv"
    echo -n "," >> "${embedding_size}.csv"
    ./galago-3.16-bin/bin/galago eval --judgments= "/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/test.qrels" --runs+ "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"  --metrics+recip_rank --metrics+ndcg20 | awk 'NR==2{print $3}' >> "${embedding_size}.csv"
  done
done