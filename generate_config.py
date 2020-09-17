import os

LSE_template = '''arch:
  input_feed: "esrt.input_feed.LSEInputFeed"
  learning_algorithm: "esrt.models.LSE"
  dataset_type: "esrt.dataset.LSEDataset"
data:
  data_dir: "/home/share/yinxiangkun/transformed/cold_start/{task}/{dataset}/seq_min_count5/"
  input_train_dir: "/home/share/yinxiangkun/transformed/cold_start/{task}/{dataset}/seq_min_count5/seq_query_split/"
  model_dir: "/home/share/yinxiangkun/saved/cold_start/{task}/lse_{embedding_size}_{dataset}/"
  logging_dir: "/home/share/yinxiangkun/log/cold_start/{task}/lse_{embedding_size}_{dataset}/"

experiment:
  subsampling_rate: 0.0001
  max_train_epoch: 50
  steps_per_checkpoint: 200
  seconds_per_checkpoint: 3600
  decode: false
  test_mode: "product_scores"
  rank_cutoff: 100

hparams:
  window_size: 5
  embed_size: 100
  max_gradient_norm: 5.0
  init_learning_rate: 0.5
  L2_lambda: 0.005
  query_weight: 0.5
  negative_sample: 5
  net_struct: "LSE"
  similarity_func: "bias_product"
  batch_size: 384'''

HEM_template = '''arch:
  input_feed: "esrt.input_feed.HEMInputFeed"
  learning_algorithm: "esrt.models.HEM"
  dataset_type: "esrt.dataset.HEMDataset"
data:
  data_dir: "/home/share/yinxiangkun/transformed/cold_start/{task}/{dataset}/seq_min_count5/"
  input_train_dir: "/home/share/yinxiangkun/transformed/cold_start/{task}/{dataset}/seq_min_count5/seq_query_split/"
  model_dir: "/home/share/yinxiangkun/saved/cold_start/{task}/hem_{embedding_size}_{dataset}/"
  logging_dir: "/home/share/yinxiangkun/log/cold_start/{task}/hem_{embedding_size}_{dataset}/"

experiment:
  subsampling_rate: 0.0001
  max_train_epoch: 50
  steps_per_checkpoint: 200
  seconds_per_checkpoint: 3600
  decode: false
  test_mode: "product_scores"
  rank_cutoff: 100

hparams:
  window_size: 5
  embed_size: 100
  max_gradient_norm: 5.0
  init_learning_rate: 0.5
  L2_lambda: 0.005
  query_weight: 0.5
  negative_sample: 5
  net_struct: "simplified_fs"
  similarity_func: "bias_product"
  batch_size: 384
'''

AEM_template = '''arch:
  input_feed: "esrt.input_feed.AEMInputFeed"
  learning_algorithm: "esrt.models.AEM"
  dataset_type: "esrt.dataset.AEMDataset"
data:
  data_dir: "/home/share/yinxiangkun/transformed/cold_start/{task}/{dataset}/seq_min_count5/"
  input_train_dir: "/home/share/yinxiangkun/transformed/cold_start/{task}/{dataset}/seq_min_count5/seq_query_split/"
  model_dir: "/home/share/yinxiangkun/saved/cold_start/{task}/aem_{embedding_size}_{dataset}/"
  logging_dir: "/home/share/yinxiangkun/log/cold_start/{task}/aem_{embedding_size}_{dataset}/"

experiment:
  subsampling_rate: 0.0001
  max_train_epoch: 50
  steps_per_checkpoint: 200
  seconds_per_checkpoint: 3600
  decode: false
  test_mode: "product_scores"
  rank_cutoff: 100

hparams:
  window_size: 5
  embed_size: 300
  max_gradient_norm: 5.0
  init_learning_rate: 0.5
  L2_lambda: 0.005
  query_weight: 0.5
  negative_sample: 5
  net_struct: "simplified_fs"
  similarity_func: "bias_product"
  user_struct: "asin_attention"
  num_heads: 5
  attention_func: 'default'
  max_history_length: 10
  batch_size: 384
'''

ZAM_template = '''arch:
  input_feed: "esrt.input_feed.ZAMInputFeed"
  learning_algorithm: "esrt.models.ZAM"
  dataset_type: "esrt.dataset.ZAMDataset"
data:
  data_dir: "/home/share/yinxiangkun/transformed/cold_start/{task}/{dataset}/seq_min_count5/"
  input_train_dir: "/home/share/yinxiangkun/transformed/cold_start/{task}/{dataset}/seq_min_count5/seq_query_split/"
  model_dir: "/home/share/yinxiangkun/saved/cold_start/{task}/zam_{embedding_size}_{dataset}/"
  logging_dir: "/home/share/yinxiangkun/log/cold_start/{task}/zam_{embedding_size}_{dataset}/"

experiment:
  subsampling_rate: 0.0001
  max_train_epoch: 50
  steps_per_checkpoint: 200
  seconds_per_checkpoint: 3600
  decode: false
  test_mode: "product_scores"
  rank_cutoff: 100

hparams:
  window_size: 5
  embed_size: 300
  max_gradient_norm: 5.0
  init_learning_rate: 0.5
  L2_lambda: 0.005
  query_weight: 0.5
  negative_sample: 5
  net_struct: "simplified_fs"
  similarity_func: "bias_product"
  user_struct: "asin_zero_attention"
  num_heads: 5
  attention_func: 'default'
  max_history_length: 10
  batch_size: 384
'''


def generate(task, model, dataset, embedding_size):
    config_dir = 'config'
    if not os.path.exists(os.path.join(config_dir, task, embedding_size, dataset)):
        os.makedirs(os.path.join(config_dir, task, embedding_size, dataset))

    if model == 'LSE':
        config = LSE_template.format(task=task, dataset=dataset, embedding_size=embedding_size)
    elif model == 'HEM':
        config = HEM_template.format(task=task, dataset=dataset, embedding_size=embedding_size)
    elif model == 'AEM':
        config = AEM_template.format(task=task, dataset=dataset, embedding_size=embedding_size)
    elif model == 'ZAM':
        config = ZAM_template.format(task=task, dataset=dataset, embedding_size=embedding_size)
    else:
        raise NotImplementedError('model not in the list')
    open(os.path.join(config_dir, task, embedding_size, dataset, model+'.yaml'), 'w').write(config)


def main():
    datasets = ['Musical_Instruments']
    models = ['LSE', 'HEM', 'AEM', 'ZAM']
    embedding_size = 'embed+0'
    max_users_list = [1, 3, 5]
    max_products_list = [1, 3, 5]
    # Ordinary
    for dataset in datasets:
        for model in models:
            generate('ordinary', model, dataset, embedding_size)
    # User Cold Start
    for max_products in max_products_list:
        for dataset in datasets:
            for model in models:
                generate('user_'+str(max_products), model, dataset, embedding_size)
    # Product Cold Start
    for max_users in max_users_list:
        for dataset in datasets:
            for model in models:
                generate('product_'+str(max_users), model, dataset, embedding_size)


if __name__ == '__main__':
    main()
