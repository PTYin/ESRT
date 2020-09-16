#!/bin/bash

cd ./seq_utils/TranSearch/

dataset=("Musical_Instruments")
max_users=(1 3 5)
max_products=(1 3 5)

# normal
for dataset in ${datasets[@]}; do
  echo "dataset = '${dataset}'
main_path = '/home/share/yinxiangkun/data/cold_start/'
stop_file = '../../seq_utils/TranSearch/stopwords.txt'
processed_path = '/home/share/yinxiangkun/cold_start_processed/${dataset}/'
" > config.py
  python preprocess.py --review_file "reviews_${dataset}_5.json.gz" --meta_file "meta_${dataset}.json.gz" --img_feature_file "image_features_${dataset}.b"
done