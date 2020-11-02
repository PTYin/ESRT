import os
from argparse import ArgumentParser
import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset',
                        default='Musical_Instruments',
                        help='name of the dataset')
    parser.add_argument('--processed_path',
                        default='/home/share/yinxiangkun/processed/cold_start/ordinary/Musical_Instruments/',
                        help="preprocessed path of the raw data")
    config = parser.parse_args()
    train_path = os.path.join(config.processed_path, "{}_train.csv".format(config.dataset))
    test_path = os.path.join(config.processed_path, "{}_test.csv".format(config.dataset))
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df: pd.DataFrame = pd.concat([train_df, test_df], ignore_index=True)

    train_support = full_df[full_df["metaFilter"] == "TrainSupport"]
    train_query = full_df[full_df["metaFilter"] == "TrainQuery"]
    test_support = full_df[full_df["metaFilter"] == "TestSupport"]
    test_query = full_df[full_df["metaFilter"] == "TestQuery"]
