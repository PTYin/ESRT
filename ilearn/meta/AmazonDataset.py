import json

from pandas import DataFrame
from torch.utils.data import Dataset
from gensim.models import Doc2Vec


class AmazonDataset(Dataset):
    def __init__(self, df: DataFrame, doc2vec: Doc2Vec, query_map: dict):
        """
        Parameters
        ----------
        df: DataFrame
        doc2vec: Doc2Vec
        query_map: dict
        """
        self.df = df
        self.df.reset_index(drop=True)
        self.doc2vec = doc2vec
        self.query_map = query_map

        products = df.groupby("asin")
        self.product_map = dict(zip(map(lambda product: self.product_map[product[0]], products), range(len(products))))

    def __getitem__(self, index):
        """
        Return
        ----------
        (user, item, query)
        """
        return self.df.loc[index, 'userID'],\
               self.product_map[self.df.loc[index, 'asin']],\
               self.doc2vec.docvecs[self.query_map[self.df.loc[index, 'query_']]]
        pass

    # def sample_neg(self):
