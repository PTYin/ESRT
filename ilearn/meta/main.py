import json
import os
from argparse import ArgumentParser
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from AmazonDataset import AmazonDataset
from Model import Model
from evaluate import metrics

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    parser = ArgumentParser()
    # ------------------------------------Dataset Parameters------------------------------------
    parser.add_argument('--dataset',
                        default='Musical_Instruments',
                        help='name of the dataset')
    parser.add_argument('--processed_path',
                        default='/home/yxk/share/yinxiangkun/processed/cold_start/ordinary/Musical_Instruments/',
                        help="preprocessed path of the raw data")
    # ------------------------------------Experiment Setup------------------------------------
    parser.add_argument('--device',
                        default='cuda:0',
                        help="using device(cpu, cuda:0, cuda:1, ...)")
    parser.add_argument('--epochs',
                        default=20,
                        type=int,
                        help="training epochs")
    # ------------------------------------Model Hyper Parameters------------------------------------
    parser.add_argument('--word_embedding_size',
                        default=128,
                        type=int,
                        help="word embedding size")
    parser.add_argument('--doc_embedding_size',
                        default=256,
                        type=int,
                        help="LSTM hidden size")
    parser.add_argument('--attention_hidden_dim',
                        default=384,
                        type=int,
                        help="LSTM hidden size")
    parser.add_argument('--margin',
                        default=1.,
                        type=float,
                        help="Margin Loss margin")

    # ------------------------------------Data Preparation------------------------------------
    config = parser.parse_args()
    train_path = os.path.join(config.processed_path, "{}_train.csv".format(config.dataset))
    test_path = os.path.join(config.processed_path, "{}_test.csv".format(config.dataset))

    query_path = os.path.join(config.processed_path, '{}_query.json'.format(config.dataset))
    asin_sample_path = config.processed_path + '{}_asin_sample.json'.format(config.dataset)
    word_dict_path = os.path.join(config.processed_path, '{}_word_dict.json'.format(config.dataset))

    query_dict = json.load(open(query_path, 'r'))
    asin_dict = json.load(open(asin_sample_path, 'r'))
    word_dict = json.load(open(word_dict_path, 'r'))

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df: pd.DataFrame = pd.concat([train_df, test_df], ignore_index=True)

    init = AmazonDataset.init(full_df)

    train_support = full_df[full_df["metaFilter"] == "TrainSupport"]
    train_query = full_df[full_df["metaFilter"] == "TrainQuery"]
    test_support = full_df[full_df["metaFilter"] == "TestSupport"]
    test_query = full_df[full_df["metaFilter"] == "TestQuery"]

    train_dataset = AmazonDataset(train_support, train_query, train_df, query_dict, asin_dict, config.device)
    test_dataset = AmazonDataset(test_support, test_query, train_df, query_dict, asin_dict, config.device)
    train_loader = DataLoader(train_dataset, drop_last=True, batch_size=3, shuffle=True, num_workers=0,
                              collate_fn=AmazonDataset.collect_fn)
    # valid_loader = DataLoader(valid_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=AmazonDataset.collect_fn)

    # ------------------------------------Model Construction------------------------------------
    # word_dict starts from 1
    model = Model(len(word_dict)+1, config.word_embedding_size, config.doc_embedding_size, config.attention_hidden_dim)
    model.to(config.device)

    criterion = nn.TripletMarginLoss(margin=config.margin, eps=1e-4)
    local_optimizer = optim.Adam(model.local_parameters, lr=0.01)
    global_optimizer = optim.Adam(model.parameters(), lr=0.01)

    # ------------------------------------Train------------------------------------
    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        loss = 0  # No effect, ignore this line
        for _, (user_reviews_words, user_reviews_lengths,
                support_item_reviews_words, support_item_reviews_lengths, support_queries,
                support_negative_reviews_words, support_negative_reviews_lengths,
                query_item_reviews_words, query_item_reviews_lengths, query_queries,
                query_negative_reviews_words, query_negative_reviews_lengths, _) in enumerate(train_loader):
            # model.train()

            # ---------Local Update---------
            model.zero_grad()
            model.set_local()
            for i in range(len(support_item_reviews_words)):
                # ---------Construct Batch---------

                pred, pos, neg = model(user_reviews_words, user_reviews_lengths,
                                       support_item_reviews_words[i], support_item_reviews_lengths[i],
                                       support_queries[i], 'train',
                                       support_negative_reviews_words[i], support_negative_reviews_lengths[i])
                loss = criterion(pred, pos, neg)
                loss.backward()

                local_optimizer.step()

            # ---------Global Update---------
            model.zero_grad()
            model.set_global()
            for i in range(len(query_item_reviews_words)):
                # ---------Construct Batch---------

                pred, pos, neg = model(user_reviews_words, user_reviews_lengths,
                                       query_item_reviews_words[i], query_item_reviews_lengths[i],
                                       query_queries[i], 'train',
                                       query_negative_reviews_words[i], query_negative_reviews_lengths[i])
                loss = criterion(pred, pos, neg)
                loss.backward()
                global_optimizer.step()

        Mrr, Hr, Ndcg = metrics(model, test_dataset, test_loader, 20, local_optimizer, criterion)
        print(
            "Running Epoch {:03d}/{:03d}".format(epoch + 1, config.epochs),
            "loss:{:.3f}".format(float(loss)),
            "Mrr {:.3f}, Hr {:.3f}, Ndcg {:.3f}".format(Mrr, Hr, Ndcg),
            "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))

    print(model.local_parameters)

