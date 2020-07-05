# Transform

## Input

- train: reviewerID,productID,reviewText,query
- test: 同上

## Objective Files

- output_path + 'train.txt.gz'
- output_path + 'test.txt.gz'

以上两个文件分别是训练集和测试集的review文本信息，每行开头是评论的user序号，接着一个制表符，再跟着评论的product序号，再制表符，后面就是review中的word序号

- output_path + 'train_id.txt.gz'
- output_path + 'test_id.txt.gz'

以上两个文件分别是训练集和测试集的review id信息，每行开头是评论的user序号，接着一个制表符，再跟着评论的product序号，再制表符，后面就是review的序号（由review.txt.gz指明）

- output_path + 'query.txt.gz'：训练集和测试集中所有的query，每行空格分隔开word（vocab.txt.gz指定）
- output_path + 'train_query_idx.txt.gz'：训练集中query的id，每行对应一个product（行号与product.txt.gz行号对应），每行若干个query（query.txt.gz指定）
- output_path + 'test_query_idx.txt.gz'：同上，但为测试集

以上用于模型输入

- output_path + 'test.qrels'
- output_path + 'test_query.json'
- output_path + 'train.qrels'
- output_path + 'train_query.json'

以上用于利用galago进行evaluation

## Algorithm

- output_path + 'train.txt.gz'
- output_path + 'test.txt.gz'

将csv文件中的reviewerID提取出来在users.txt.gz中查找对应序号，asin在product.txt.gz中查找对应序号，reviewText分别在vocab.txt.gz中查找对应序号

- output_path + 'train_id.txt.gz'
- output_path + 'test_id.txt.gz'

所有模型均不需要这两个文件

