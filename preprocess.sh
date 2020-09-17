#!/bin/bash

cd ./seq_utils/TranSearch/

datasets=("Musical_Instruments")
max_users_list=(1 3 5)
max_products_list=(1 3 5)

# ordinary
for dataset in ${datasets[@]}; do
  python preprocess.py --review_file "reviews_${dataset}_5.json.gz" --meta_file "meta_${dataset}.json.gz" --img_feature_file "image_features_${dataset}.b" --dataset "${dataset}" --processed_path "/home/share/yinxiangkun/processed/cold_start/ordinary/${dataset}/"
  python doc2vec_img2dict.py --img_feature_file "image_features_${dataset}.b" --dataset "${dataset}" --processed_path "/home/share/yinxiangkun/processed/cold_start/ordinary/${dataset}/"
  python transform.py --csv_dir "/home/share/yinxiangkun/processed/cold_start/ordinary/${dataset}/" --review_file "/home/share/yinxiangkun/data/cold_start/reviews_${dataset}_5.json.gz" --output_dir "/home/share/yinxiangkun/transformed/cold_start/ordinary/${dataset}/" --meta_file "/home/share/yinxiangkun/data/cold_start/reviews_${dataset}_5.json.gz" --dataset "${dataset}"
done

# user cold start
for max_products in ${max_products_list[@]}; do
  for dataset in ${datasets[@]}; do
    python preprocess.py --review_file "reviews_${dataset}_5.json.gz" --meta_file "meta_${dataset}.json.gz" --img_feature_file "image_features_${dataset}.b" --max_products_per_user ${max_products} --dataset "${dataset}" --processed_path "/home/share/yinxiangkun/processed/cold_start/user_${max_products}/${dataset}/"
    python doc2vec_img2dict.py --img_feature_file "image_features_${dataset}.b" --dataset "${dataset}" --processed_path "/home/share/yinxiangkun/processed/cold_start/user_${max_products}/${dataset}/"
    python transform.py --csv_dir "/home/share/yinxiangkun/processed/cold_start/user_${max_products}/${dataset}/" --review_file "/home/share/yinxiangkun/data/cold_start/reviews_${dataset}_5.json.gz" --output_dir "/home/share/yinxiangkun/transformed/cold_start/user_${max_products}/${dataset}/" --meta_file "/home/share/yinxiangkun/data/cold_start/reviews_${dataset}_5.json.gz" --dataset "${dataset}"
  done
done


# product cold start
for max_users in ${max_users_list[@]}; do
  for dataset in ${datasets[@]}; do
    python preprocess.py --review_file "reviews_${dataset}_5.json.gz" --meta_file "meta_${dataset}.json.gz" --img_feature_file "image_features_${dataset}.b" --max_users_per_product ${max_users} --dataset "${dataset}" --processed_path "/home/share/yinxiangkun/processed/cold_start/product_${max_users}/${dataset}/"
    python doc2vec_img2dict.py --img_feature_file "image_features_${dataset}.b" --dataset "${dataset}" --processed_path "/home/share/yinxiangkun/processed/cold_start/product_${max_users}/${dataset}/"
    python transform.py --csv_dir "/home/share/yinxiangkun/cold_start/processed/product_${max_users}/${dataset}/" --review_file "/home/share/yinxiangkun/data/cold_start/reviews_${dataset}_5.json.gz" --output_dir "/home/share/yinxiangkun/transformed/cold_start/product_${max_users}/${dataset}/" --meta_file "/home/share/yinxiangkun/data/cold_start/reviews_${dataset}_5.json.gz" --dataset "${dataset}"
  done
done
