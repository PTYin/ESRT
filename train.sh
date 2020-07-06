
dataset=("automotive" "cellphones" "clothing" "digital_music" "electronics" "toys")
models=("lse" "hem" "aem" "zam")
embedding_size="embed+0"

for (( i = 0; i < 6; i++ )); do
  for (( j = 0; j < 4; j++ )); do
    python main.py --setting_file "config/${dataset[i]}/${models[j]}_exp.yaml" >> "/home/share/yinxiangkun/log/${embedding_size}/${dataset[i]}/${models}/train_log.txt"
    python main.py --setting_file "config/${dataset[i]}/${models[j]}_exp.yaml" --decode True >> "/home/share/yinxiangkun/log/${embedding_size}/${dataset[i]}/${models}/test_log.txt"
    cp "/home/share/yinxiangkun/saved/${models[j]}_${embedding_size}_${dataset[i]}/test.bias_product.ranklist" "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"
#    ./galago-3.16-bin/bin/galago eval --judgments= "/home/share/yinxiangkun/transformed/${dataset[i]}/seq_min_count5/seq_query_split/test.qrels" --runs+ "/home/share/yinxiangkun/saved/${embedding_size}/${dataset[i]}/${models[j]}.ranklist"  --metrics+recip_rank --metrics+ndcg10 --metrics+P10 >> ./log.txt
  done
done

