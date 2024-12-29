import os

import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from fwc2.utils import load_data, clearn_data, tsne_scatter

SEED = 42


def main(ds_name: str, task: str = 'anomaly', train_size: float = 0.8):
    print(f'*** dataset = {ds_name} ')

    data = load_data(ds_name, raw=True, base_dir='./datasets')
    cleaned = clearn_data(data, impute_nan=True)

    if task == 'anomaly':
        b_mask = cleaned['label'] == 'benign'
        benign_data = cleaned[b_mask]
        train_set = benign_data.sample(frac=train_size, random_state=SEED)
        test_set = pd.concat([benign_data.drop(train_set.index), cleaned[~b_mask]], ignore_index=True)
    else:
        train_set, test_set = train_test_split(cleaned, train_size=train_size, stratify=cleaned['label'], random_state=SEED)


    print('** train_set distribution')
    print(train_set['label'].value_counts())
    print('** test_set distribution')
    print(test_set['label'].value_counts())

    our_dir = f'./data/{ds_name}/{task}'
    os.makedirs(our_dir, exist_ok=True)

    train_set.to_csv(os.path.join(our_dir, 'train.csv'), index=False, encoding='utf-8')
    test_set.to_csv(os.path.join(our_dir, 'test.csv'), index=False, encoding='utf-8')

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--ds", type=str, default='dapt20')
    parser.add_argument("--task", type=str, default='anomaly')
    parser.add_argument("--train-s", type=float, default=0.8)

    args = parser.parse_args()

    main(args.ds, args.task, args.train_s)
