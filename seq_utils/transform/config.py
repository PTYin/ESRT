# choose dataset name
#dataset = 'Clothing_Shoes_and_Jewelry'
dataset = 'Automotive'

# paths
main_path = '/home/share/yinxiangkun/automotive/'
stop_file = './stopwords.txt'

processed_path = '/home/share/yinxiangkun/automotive/processed/'

full_path = main_path + '{}_full.csv'.format(dataset)
train_path = main_path + '{}_train.csv'.format(dataset)
test_path = main_path + '{}_test.csv'.format(dataset)

asin_sample_path = main_path + '{}_asin_sample.json'.format(dataset)
user_bought_path = main_path + '{}_user_bought.json'.format(dataset)

doc2model_path = main_path + '{}_doc2model'.format(dataset)
query_path = main_path + '{}_query.json'.format(dataset)

# embedding size
embed_size = 16
