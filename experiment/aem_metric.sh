#./galago-3.16-bin/bin/galago eval --judgments= ../amazon_cellphone_index_dataset/cell_phone/random_query_split/test.qrels --runs+ ../hem_tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10
cd ..

# download galago 
if ! [ -d "./galago-3.16-bin/" ]; then
  wget https://iweb.dl.sourceforge.net/project/lemur/lemur/galago-3.16/galago-3.16-bin.tar.gz
  tar xvzf galago-3.16-bin.tar.gz
fi

# evaluate AEM
./galago-3.16-bin/bin/galago eval --judgments= /home/share/yinxiangkun/indexed_data/seq_min_count5/seq_query_split/test.qrels --runs+ ./aem_tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10


