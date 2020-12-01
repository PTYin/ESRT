#!/bin/bash
datasets=("Musical_Instruments")
embedding_size="embed+0"
task=$1 # ('ordinary' 'user_1' 'user_3' 'user_5' 'product_1' 'product_3' 'product_5')
cd ./ilearn/TranSearch_without_images

for dataset in ${datasets[@]}; do
    if [ ! -d "/home/yxk/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/${dataset[i]}/TranSearch/" ]; then
      mkdir "/home/yxk/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/${dataset[i]}/TranSearch/"
    fi
  python run.py --dataset ${dataset} --processed_path /home/yxk/share/yinxiangkun/processed/cold_start/${task}/${dataset}/ >> "/home/yxk/share/yinxiangkun/log/cold_start/${task}/${embedding_size}/${dataset[i]}/TranSearch/train_log.txt"
done