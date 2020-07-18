import argparse
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--judgments',
                        type=str,
                        default='./data/train.qrels')
    parser.add_argument('--ranklist',
                        type=str,
                        default='./data/test.bias_product.ranklist')
    parser.add_argument('--cutoff',
                        type=int,
                        default=20)
    flags = parser.parse_args()
    cutoff = flags.cutoff
    user_bought = {}
    hits = 0
    total = 0
    with open(flags.judgments, 'r') as standard, open(flags.ranklist, 'r') as result:
        for line in standard:
            user = re.sub(r'_[0-9]+', '', line.split(' ')[0])
            if user not in user_bought:
                user_bought[user] = set()
            user_bought[user].add(line.split(' ')[2])
            total += 1
        for line in result:
            user = re.sub(r'_[0-9]+', '', line.split(' ')[0])
            item = line.split(' ')[2]
            rank = int(line.split(' ')[3])
            if rank <= cutoff and item in user_bought[user]:
                hits += 1
    print(round(hits/total, 3))
