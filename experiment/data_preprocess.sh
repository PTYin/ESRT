# Download Amazon review dataset "Cell_Phones_and_Accessories" 5-core.
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz

# Download the meta data from http://jmcauley.ucsd.edu/data/amazon/
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Cell_Phones_and_Accessories.json.gz

# Stem and remove stop words from the Amazon review datasets if needed. Here, we stem the field of “reviewText” and “summary” without stop words removal.
#java -Xmx4g -jar ./seq_utils/AmazonDataset/jar/AmazonReviewData_preprocess.jar false ./reviews_Cell_Phones_and_Accessories_5.json.gz ./reviews_Cell_Phones_and_Accessories_5.processed.gz

# Index dataset
python ../seq_utils/AmazonDataset/index_and_filter_review_file.py /home/share/yinxiangkun/reviews_Musical_Instruments_5.json.gz /home/share/yinxiangkun/indexed_data/ 5

# Match the meta data with the indexed data to extract queries:
java -Xmx16G -jar ../seq_utils/AmazonDataset/jar/AmazonMetaData_matching.jar false /home/share/yinxiangkun/metadata.json.gz /home/share/yinxiangkun/indexed_data/seq_min_count5/

# Gather knowledge from meta data:
python ../seq_utils/AmazonDataset/match_with_meta_knowledge.py /home/share/yinxiangkun/indexed_data/seq_min_count5/ /home/share/yinxiangkun/metadata.json.gz

# Sequentially split train/test
python ./seq_utils/AmazonDataset/sequentially_split_train_test_data.py /home/share/yinxiangkun/indexed_data/seq_min_count5/ 0.3 0.3


