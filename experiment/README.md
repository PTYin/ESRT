# Data Preprocess

## 1. Index and filter review

index_and_filter_review_file.py

### Arguments

1. review_file = sys.argv[1]
2. output_path = sys.argv[2]
3. min_count = int(sys.argv[3])

第三个参数是用来筛选word的，如果review中的一个word出现次数小于min_count，那么过滤掉

### Output

- output_path + 'vocab.txt.gz' 
- output_path + 'users.txt.gz'
- output_path + 'product.txt.gz'

将所有的word,user_id,product_id分别输出到以上三个文件，索引文件

- output_path + 'review_text.txt.gz'：各个词的序号，一行是一个review的词（过滤过）
- output_path + 'review_u_p.txt.gz'：每行两个序号，分别对应该行评论的user序号和product序号
- output_path + 'review_id.txt.gz'：顾名思义，review的id，实际上就是在每行记录一下其在review_file中的序号（因为可能存在某些review没有word）
- output_path + 'review_rating.txt.gz'：每行对应该行review给出的评分

以上文件用于HEM

- output_path + 'u_r_seq.txt.gz'：每行对应一个user的所有review，按照时间升序排序
- output_path + 'review_loc_and_time.txt.gz'：每行对应一个review，行中第一个序号代表的是该review是对应user评论的第几条review（时间顺序），行中第二个数字是review的unix时间

## 2. Match the meta data with the indexed data to extract queries

AmazonMetaData_matching.jar

### Arguments

1. \<jsonConfigFile\>: A json file that specify the file path of stop words list. An example can be found in the root directory. Enter “false” if don’t want to remove stop words. 
2. \<review_file\>: the path for the original Amazon review data
3. \<output_review_file\>: the output path for processed Amazon review data

### Algorithm
首先过滤掉非字母字符，之后用galago的KrovetzStemmer进行词干提取

### Output

- indexed_review_path + "product_des.txt.gz"：通过提取metadata中的"title"/"description"两个field中的词获得产品的description，每一行代表一个产品，每行的数字是word的序号
- indexed_review_path + "product_query.txt.gz"：通过提取metadata中的"category"field中的词获得query，每一行代表一个产品，开头c+数字，数字代表该产品subcategory的数量，制表符后面的数字是word的序号

## 3. Gather knowledge from meta data

match_with_meta_knowledge.py

### Arguments

1.	data_path = sys.argv[1]
2.	meta_path = sys.argv[2]

第一个参数指示前序步骤产生的文件所在目录，第二个参数指示metadata.json.gz的路径

### Output

- data_path + 'also_bought_p_p.txt.gz'：一行多个序号，分别代表和该行product具有also_bought关系的product序号，若没有该行空白（注意这个序号并不是product.txt.gz中的product序号而是related_product.txt.gz中的从0开始行号）
- data_path + 'also_viewed_p_p.txt.gz'：同上，具有also_viewed关系product序号
- data_path + 'bought_together_p_p.txt.gz'：同上，具有bought_together关系product序号
- data_path + 'brand_p_b.txt.gz'：一行中的序号代表该行product所属的brand序号，若没有该行空白（brand序号是brand.txt.gz中的行号）
- data_path + 'category_p_c.txt.gz'：一行中的序号代表该行product所属的category序号（category序号是category.txt.gz中的行号）

以上是模型用的

- data_path + 'related_product.txt.gz'：记录了在metadata.json中具有"also_bought"/"also_viewed"/"bought_together"中至少一个标签的product
- data_path + 'brand.txt.gz'：记录了所有的brand
- data_path + 'category.txt.gz'：记录了所有的category
- data_path + 'knowledge_statistics.txt'：记录了一些统计信息

以上是索引文件

## 4. Sequentially split train/test

sequentially_split_train_test_data.py

### Arguments

1. data_path = sys.argv[1]
2. review_sample_rate = float(sys.argv[2])  # Percentage of reviews used for test for each user
3. query_sample_rate = float(sys.argv[3])  # Percentage of queries that are unique in testing

第二个和第三个参数分别指示对于每个用户用于测试的reviews比例、在测试中没有在训练集出现过的query比例

### Output

- output_path + 'train.txt.gz'
- output_path + 'test.txt.gz'
- 