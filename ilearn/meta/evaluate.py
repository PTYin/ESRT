import numpy as np

import torch
import torch.nn.functional as F

from AmazonDataset import AmazonDataset

def mrr(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(float(index + 1))
    else:
        return 0


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_dataset, test_loader, top_k, local_optimizer, criterion):
    Mrr, Hr, Ndcg = [], [], []
    loss = 0  # No effect, ignore this line
    for _, (user_reviews_words, user_reviews_lengths,
            support_item_reviews_words, support_item_reviews_lengths, support_queries,
            support_negative_reviews_words, support_negative_reviews_lengths,
            query_item_reviews_words, query_item_reviews_lengths, query_queries,
            query_negative_reviews_words, query_negative_reviews_lengths, query_item_asin) in enumerate(test_loader):

        # ---------Local Update---------
        model.train()
        model.zero_grad()
        model.set_local()
        for i in range(len(support_item_reviews_words)):
            # ---------Construct Batch---------

            pred, pos, neg = model(user_reviews_words, user_reviews_lengths,
                                   support_item_reviews_words[i], support_item_reviews_lengths[i],
                                   support_queries[i], 'train', support_negative_reviews_words[i],
                                   support_negative_reviews_lengths[i],)
            loss = criterion(pred, pos, neg)
            loss.backward()

            local_optimizer.step()

        # ---------Global Update---------
        model.eval()
        assert len(query_item_reviews_words) == 1
        pred, pos = model(user_reviews_words, user_reviews_lengths,
                          query_item_reviews_words[0], query_item_reviews_lengths[0],
                          query_queries[0], 'test')
        candidates_reviews_words, candidates_reviews_lengths = test_dataset.neg_candidates(query_item_asin)
        candidates = []
        for i in range(99):
            # ---------Construct Batch---------
            candidates.append(model(user_reviews_words, user_reviews_lengths,
                                    candidates_reviews_words[i], candidates_reviews_lengths[i],
                                    query_queries[0], 'output_embedding'))
        candidates = [pos] + candidates
        candidates = torch.stack(candidates, dim=0)

        scores = F.pairwise_distance(pred.repeat(100, 1), candidates)
        _, ranking_list = scores.sort(dim=-1, descending=True)
        ranking_list = ranking_list.tolist()
        top_idx = []
        while len(top_idx) < top_k:
            candidate_item = ranking_list.pop()
            top_idx.append(candidate_item)

        Mrr.append(mrr(0, top_idx))
        Hr.append(hit(0, top_idx))
        Ndcg.append(ndcg(0, top_idx))
    return np.mean(Mrr), np.mean(Hr), np.mean(Ndcg)
