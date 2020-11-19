import os
import gzip
import json
import argparse
import itertools
import random

import pandas as pd
import numpy as np

import text_process


def get_df(path):
    """ Apply raw data to pandas DataFrame. """
    idx = 0
    df = {}
    g = gzip.open(path, 'rb')
    for line in g:
        df[idx] = json.loads(line)
        idx += 1
    return pd.DataFrame.from_dict(df, orient='index')


def extraction(meta_path, review_df, stop_words, count):
    """ Extracting useful infromation. """
    with gzip.open(meta_path, 'rb') as g:
        categories, also_viewed = {}, {}
        for line in g:
            line = eval(line)
            asin = line['asin']
            categories[asin] = line['categories']
            related = line['related'] if 'related' in line else None

            # fill the also_related dictionary
            also_viewed[asin] = []
            relations = ['also_viewed', 'buy_after_viewing']
            if related:
                also_viewed[asin] = [related[r] for r in relations if r in related]
                also_viewed[asin] = itertools.chain.from_iterable(also_viewed[asin])

    queries, reviews = [], []
    word_dict = {}  # start from 1
    for i in range(len(review_df)):
        asin = review_df['asin'][i]
        review = review_df['reviewText'][i]
        category = categories[asin]

        # process queries
        qs = map(text_process._remove_dup,
                 map(text_process._remove_char, category))
        qs = [[w for w in q if w not in stop_words] for q in qs]

        for q in qs:
            for w in q:
                if w not in word_dict:
                    word_dict[w] = len(word_dict)+1

        # process reviews
        review = text_process._remove_char(review)
        review = [w for w in review if w not in stop_words]

        for w in review:
            if w not in word_dict:
                word_dict[w] = len(word_dict)+1

        queries.append(qs)
        reviews.append(review)

    review_df['query_'] = queries  # write query result to dataframe

    # filtering words counts less than count
    reviews = text_process._filter_words(reviews, count)
    review_df['reviewText'] = reviews

    review_df['reviewWords'] = [[word_dict[w] for w in review] for review in reviews]
    review_df['queryWords'] = [[[word_dict[w] for w in q] for q in qs] for qs in queries]

    return review_df, also_viewed, word_dict


def reindex(df):
    """ Reindex the reviewID from 0 to total length. """
    reviewer = df['reviewerID'].unique()
    reviewer_map = {r: i for i, r in enumerate(reviewer)}

    userIDs = [reviewer_map[df['reviewerID'][i]] for i in range(len(df))]
    df['userID'] = userIDs
    return df


def split_data(df, max_users_per_product, max_products_per_user):
    """ Enlarge the dataset with the corresponding user-query-item pairs."""
    df_enlarge = {}
    i = 0
    for row in range(len(df)):
        for j in range(len(df['query_'][row])):
            df_enlarge[i] = {'reviewerID': df['reviewerID'][row],
                             'userID': df['userID'][row], 'query_': df['query_'][row][j],
                             'queryWords': df['queryWords'][row][j],
                             'asin': df['asin'][row], 'reviewText': df['reviewText'][row],
                             'reviewWords': df['reviewWords'][row],
                             'gender': None}
            i += 1
    df_enlarge = pd.DataFrame.from_dict(df_enlarge, orient='index')

    split_filter = []
    df_enlarge = df_enlarge.sort_values(by='userID')
    df_enlarge['filter'] = None
    df_enlarge['metaFilter'] = None
    users = df_enlarge.groupby('userID')
    user_length = users.size().tolist()
    users = list(df_enlarge.groupby('userID'))

    random.shuffle(users)

    # Meta Learning Train Set
    for index in range(int(len(user_length) * 0.7)):
        user = users[index][1]
        length = len(user)
        df_enlarge.loc[user.index, 'filter'] = ['Train'] * length
        meta_tag = ['TrainSupport'] * int(length * 0.6) + ['TrainQuery'] * (length - int(length * 0.6))
        random.shuffle(meta_tag)
        df_enlarge.loc[user.index, 'metaFilter'] = meta_tag

    # Meta Learning Test Set
    for index in range(int(len(user_length) * 0.7), len(user_length)):
        user = users[index][1]
        length = len(user)

        test_pos = random.randint(0, length-1)
        df_enlarge.loc[user.index, 'filter'] = ['Train'] * test_pos + ['Test'] + ['Train'] * (length - test_pos - 1)
        df_enlarge.loc[user.index, 'metaFilter'] = \
            ['TestSupport'] * test_pos + ['TestQuery'] + ['TestSupport'] * (length - test_pos - 1)

    # for asin in df_enlarge.groupby(by='asin'):
    #     asin = asin[1]
    #     train_length = len(asin[asin['metaFilter'].startswith('Train')])
    #     train_query = asin[asin['metaFilter'] == 'TrainQuery']
    #     if len(train_query) == train_length:
    #         for user in train_query['userID']:

    # ----------------Cut for Cold Start----------------
    if max_products_per_user is not None:  # cold start for new user
        users = df_enlarge.groupby('userID')
        for products_per_user in users:
            products_per_user = products_per_user[1]
            products_per_user = products_per_user[products_per_user['filter'] == 'Train']
            if len(products_per_user) > max_products_per_user:
                df_enlarge.drop(products_per_user[max_products_per_user:].index, inplace=True)

    if max_users_per_product is not None:  # # cold start for new product
        products = df_enlarge.groupby('asin')
        for users_per_product in products:
            users_per_product = users_per_product[1]
            users_per_product = users_per_product[users_per_product['filter'] == 'Train']
            if len(users_per_product) > max_users_per_product:
                df_enlarge.drop(users_per_product[max_users_per_product:].index, inplace=True)

    df_enlarge_train = df_enlarge[df_enlarge['filter'] == 'Train']
    df_enlarge_test = df_enlarge[df_enlarge['filter'] == 'Test']
    print('---------------', len(df_enlarge))
    return (df_enlarge.reset_index(drop=True),
            df_enlarge_train.reset_index(drop=True),
            df_enlarge_test.reset_index(drop=True))


def get_user_bought(train_set):
    """ Obtain the products each user has bought before test. """
    user_bought = {}
    for i in range(len(train_set)):
        user = train_set['reviewerID'][i]
        item = train_set['asin'][i]
        if user not in user_bought:
            user_bought[user] = []
        user_bought[user].append(item)
    return user_bought


def rm_test(df, df_test):
    """ Remove test review data and remove duplicate reviews."""
    df = df.reset_index(drop=True)
    reviewText = []
    review_train_set = set()

    review_test = set(repr(
        df_test['reviewText'][i]) for i in range(len(df_test)))

    for i in range(len(df)):
        r = repr(df['reviewText'][i])
        if not r in review_train_set and not r in review_test:
            review_train_set.add(r)
            reviewText.append(df['reviewText'][i])
        else:
            reviewText.append("[]")
    df['reviewText'] = reviewText
    return df


def neg_sample(also_viewed, unique_asin):
    """
    Sample the negative set for each asin(item), first add the 'also_view'
    asins to the dict, then add asins share the same query.
    """
    asin_samples = {}
    for asin in unique_asin:
        positive = set([a for a in also_viewed[asin] if a in unique_asin])
        negative = list(unique_asin - positive)
        if not len(positive) < 20:
            negative = np.random.choice(
                negative, 5 * len(positive), replace=False).tolist()

        elif not len(positive) < 5:
            negative = np.random.choice(
                negative, 10 * len(positive), replace=False).tolist()

        elif not len(positive) < 1:
            negative = np.random.choice(
                negative, 20 * len(positive), replace=False).tolist()

        else:
            negative = np.random.choice(negative, 50, replace=False).tolist()

        pos_pr = [0.7 for _ in range(len(positive))]
        neg_pr = [0.3 for _ in range(len(negative))]
        prob = np.array(pos_pr + neg_pr)
        prob = prob / prob.sum()

        asin_samples[asin] = {'positive': list(positive),
                              'negative': negative,
                              'prob': prob.tolist()}
    return asin_samples


def filter_review(review_df):
    review_df = review_df.drop(index=review_df[review_df.reviewText.map(len) == 0].index)
    users = review_df.groupby('reviewerID')
    for interactions in users:
        interactions = interactions[1]
        if len(interactions) < 2:
            review_df.drop(interactions.index)
    return review_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review_file',
                        type=str,
                        default='reviews_Musical_Instruments_5.json.gz',
                        help="5-core review file")
    parser.add_argument('--meta_file',
                        type=str,
                        default='meta_Musical_Instruments.json.gz',
                        help="meta data file for the corresponding review file")
    parser.add_argument('--count',
                        type=int,
                        default=5,
                        help="remove the words number less than count")
    parser.add_argument("--max_users_per_product",
                        type=int,
                        default=None,
                        help="define the maximum number of bought products per user, no maximum number if None")
    parser.add_argument("--max_products_per_user",
                        type=int,
                        default=None,
                        help="define the maximum number of users per product, no maximum number if None")

    parser.add_argument('--dataset', type=str, default='Musical_Instruments')
    parser.add_argument('--main_path', type=str, default='/home/share/yinxiangkun/data/cold_start/')
    parser.add_argument('--stop_file', type=str, default='../../seq_utils/TranSearch/stopwords.txt')
    parser.add_argument('--processed_path', type=str,
                        default='/home/share/yinxiangkun/processed/cold_start/ordinary/Musical_Instruments/')

    global FLAGS
    FLAGS = parser.parse_args()
    if (FLAGS.max_users_per_product is not None and FLAGS.max_users_per_product < 1) or\
            (FLAGS.max_products_per_user is not None and FLAGS.max_products_per_user < 1):
        raise Exception('Too few samples to train! Increase max users or max products.')

    # --------------PREPARE PATHS--------------
    if not os.path.exists(FLAGS.processed_path):
        os.makedirs(FLAGS.processed_path)

    stop_path = FLAGS.stop_file
    meta_path = os.path.join(FLAGS.main_path, FLAGS.meta_file)
    review_path = os.path.join(FLAGS.main_path, FLAGS.review_file)

    review_df = get_df(review_path)

    # --------------PRE-EXTRACTION--------------
    stop_df = pd.read_csv(stop_path, header=None, names=['stopword'])
    stop_words = set(stop_df['stopword'].unique())
    df, also_viewed, word_dict = extraction(meta_path, review_df, stop_words, FLAGS.count)
    df = df.drop(['reviewerName', 'reviewTime', 'helpful', 'summary',
                  'unixReviewTime', 'overall'], axis=1)  # remove non-useful keys

    # at least 2 interactions & filter blank review
    df = filter_review(df)

    dataset = [(FLAGS.dataset, df)]

    for d in dataset:
        df = reindex(d[1])  # reset the index of users
        df, df_train, df_test = split_data(df,
                                           max_users_per_product=FLAGS.max_users_per_product,
                                           max_products_per_user=FLAGS.max_products_per_user)

        print("The number of {} users is {:d}; items is {:d}; feedbacks is {:d}.".format(
            d[0], len(df.reviewerID.unique()), len(df.asin.unique()), len(df)))

        # sample negative items
        asin_samples = neg_sample(also_viewed, set(df.asin.unique()))
        print("Negtive samples of {} set done!".format(d[0]))

        # ---------------------------------Save Parameters---------------------------------
        json.dump(asin_samples, open(os.path.join(
            FLAGS.processed_path, '{}_asin_sample.json'.format(d[0])), 'w'))

        user_bought = get_user_bought(df_train)
        json.dump(user_bought, open(os.path.join(
            FLAGS.processed_path, '{}_user_bought.json'.format(d[0])), 'w'))

        json.dump(word_dict, open(os.path.join(
            FLAGS.processed_path, '{}_word_dict.json'.format(d[0])), 'w'))

        df = rm_test(df, df_test)  # remove the reviews from test set
        df_train = rm_test(df_train, df_test)

        df.to_csv(os.path.join(
            FLAGS.processed_path, '{}_full.csv'.format(d[0])), index=False)
        df_train.to_csv(os.path.join(
            FLAGS.processed_path, '{}_train.csv'.format(d[0])), index=False)
        df_test.to_csv(os.path.join(
            FLAGS.processed_path, '{}_test.csv'.format(d[0])), index=False)


if __name__ == "__main__":
    main()
