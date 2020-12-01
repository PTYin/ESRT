import json
import torch
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import ast


class AmazonDataset(Dataset):
    def __init__(self, support_df: DataFrame, query_df: DataFrame, train_df: DataFrame,
                 query_map: dict, asin_dict: dict, device,
                 pre_train=False):
        """
        Parameters
        ----------
        support_df: DataFrame
        query_df: DataFrame
        query_map: dict
        asin_dict: dict
        """
        self.support_df = support_df
        self.query_df = query_df
        self.train_df = train_df
        self.support_df = self.support_df.set_index(['userID'])
        self.query_df = self.query_df.set_index(['userID'])
        self.query_map = query_map
        self.asin_dict = asin_dict
        self.device = device
        self.pre_train = pre_train

        self.users = self.support_df.index.unique()

        self.user_reviews_words = {}  # record the support user-item reviews
        self.item_reviews_words = {}  # same as above

        self.user_reviews_lengths = {}  # record the corresponding lengths of users' reviews
        self.item_reviews_lengths = {}  # record the corresponding lengths of items' reviews

        self.cluster_reviews()

    def cluster_reviews(self):
        users = self.support_df.groupby(by="userID")  # for user, cluster the reviews from support data
        items = self.train_df.groupby(by="asin")  # for item, cluster all the reviews except for the test query data
        for user in users:
            user_reviews_lengths = torch.LongTensor(user[1]['reviewWords'].map(
                lambda review: len(eval(review))).tolist())
            self.user_reviews_lengths[user[0]] = user_reviews_lengths

            user_reviews_words = user[1]['reviewWords'].map(lambda review: torch.LongTensor(eval(review))).tolist()
            self.user_reviews_words[user[0]] = pad_sequence(user_reviews_words, batch_first=True, padding_value=0).to(self.device)
        for item in items:
            item_reviews_lengths = torch.LongTensor(item[1]['reviewWords'].map(
                lambda review: len(eval(review))).tolist())
            self.item_reviews_lengths[item[0]] = item_reviews_lengths

            item_reviews_words = item[1]['reviewWords'].map(lambda review: torch.LongTensor(eval(review))).tolist()
            self.item_reviews_words[item[0]] = pad_sequence(item_reviews_words, batch_first=True, padding_value=0).to(self.device)
            # shape: (batch, seq_lens)

    def __len__(self):
        return len(self.users)

    def __get_instance(self, asin,
                       item_reviews_words, item_reviews_lengths, negative_reviews_words, negative_reviews_lengths):
        item_reviews_words.append(self.item_reviews_words[asin])
        item_reviews_lengths.append(self.item_reviews_lengths[asin])
        negative_item = self.sample_neg(asin)
        negative_reviews_words.append(self.item_reviews_words[negative_item])
        negative_reviews_lengths.append(self.item_reviews_lengths[negative_item])

    def __getitem__(self, index):
        """
        Return
        ----------
        (user_reviews_words, user_reviews_lengths,
         support_item_reviews_words, support_item_reviews_lengths, support_queries,
         support_negative_reviews_words, support_negative_reviews_lengths,
          query_item_reviews_words, query_item_reviews_lengths, query_queries,
          query_negative_reviews_words, query_negative_reviews_lengths)
        """
        user = self.users[index]

        (support_item_reviews_words,
         support_item_reviews_lengths,
         support_negative_reviews_words,
         support_negative_reviews_lengths) = [], [], [], []

        (query_item_reviews_words,
         query_item_reviews_lengths,
         query_negative_reviews_words,
         query_negative_reviews_lengths) = [], [], [], []

        pd.Series(self.support_df.loc[user, 'asin'], dtype=str).apply(
            self.__get_instance, args=(support_item_reviews_words,
                                       support_item_reviews_lengths,
                                       support_negative_reviews_words,
                                       support_negative_reviews_lengths))
        support_queries = pd.Series(self.support_df.loc[user, 'queryWords'], dtype=str).map(
            lambda query: torch.LongTensor(eval(query)).to(self.device)).tolist()

        query_item_asin = self.query_df.loc[user, 'asin']  # useful iff TestQuery
        pd.Series(self.query_df.loc[user, 'asin'], dtype=str).apply(
            self.__get_instance, args=(query_item_reviews_words,
                                       query_item_reviews_lengths,
                                       query_negative_reviews_words,
                                       query_negative_reviews_lengths))
        query_queries = pd.Series(self.query_df.loc[user, 'queryWords'], dtype=str).map(
            lambda query: torch.LongTensor(eval(query)).to(self.device)).tolist()

        return (self.user_reviews_words[user], self.user_reviews_lengths[user],
                support_item_reviews_words, support_item_reviews_lengths, support_queries,
                support_negative_reviews_words, support_negative_reviews_lengths,
                query_item_reviews_words, query_item_reviews_lengths, query_queries,
                query_negative_reviews_words, query_negative_reviews_lengths, query_item_asin)

    def sample_neg(self, item):
        """ Take the also_view or buy_after_viewing as negative samples. """
        # We tend to sample negative ones from the also_view and
        # buy_after_viewing items, if don't have enough, we then
        # randomly sample negative ones.

        sample = self.asin_dict[item]
        all_sample = sample['positive'] + sample['negative']
        neg = np.random.choice(all_sample, 1, replace=False, p=sample['prob'])
        if neg[0] not in self.item_reviews_words:
            neg = np.random.choice(list(self.item_reviews_words.keys()), 1, replace=False)
        return neg[0]

    def neg_candidates(self, item):
        """random select 99 candidates to participate test evaluation"""
        a = list(self.item_reviews_words.keys() - {item, })
        candidates = np.random.choice(a, 99, replace=False)
        candidates_reviews_words = list(map(lambda candidate: self.item_reviews_words[candidate], candidates))
        candidates_reviews_lengths = list(map(lambda candidate: self.item_reviews_lengths[candidate], candidates))
        return candidates_reviews_words, candidates_reviews_lengths

    @staticmethod
    def init(full_df: DataFrame):
        users = full_df.groupby("userID")
        items = full_df.groupby("asin")
        # item_map = dict(zip(map(lambda item: item[0], items), range(len(items))))
        return {'users': users, 'items': items}

    @staticmethod
    def collect_fn(batch):

        for record in batch:
            (batch_user_reviews_words, batch_user_reviews_lengths,
             batch_support_item_reviews_words, batch_support_item_reviews_lengths, batch_support_queries,
             batch_support_negative_reviews_words, batch_support_negative_reviews_lengths,
             batch_query_item_reviews_words, batch_query_item_reviews_lengths, batch_query_queries,
             batch_query_negative_reviews_words, batch_query_negative_reviews_lengths, batch_query_item_asin) = record


        return batch[0]

    # @staticmethod
    # def construct_batch(i, batch_users, batch_items: list, batch_queries: list):
    #     users = []
    #     items = []
    #     queries = []
    #     for k in range(len(batch_users)):
    #         if i < len(batch_items[k]):
    #             users.append(batch_users[k])
    #             items.append(batch_items[k][i])
    #             queries.append(batch_queries[k][i])
    #     return torch.LongTensor(users).to(self.device), torch.LongTensor(items).to(self.device), torch.LongTensor(queries).to(self.device)
