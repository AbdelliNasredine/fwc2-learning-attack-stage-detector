import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_BASE_DIR = "../datasets"


def load_data(dataset_name: str, base_dir: str = DEFAULT_BASE_DIR, raw: bool = True) -> pd.DataFrame:
    if not raw:
        print(f'loading {dataset_name} from {base_dir}')
        return pd.read_csv(os.path.join(base_dir, dataset_name, 'clean.csv'), encoding='utf-8')

    base_dir = os.path.join(base_dir, dataset_name, 'raw')
    assert os.path.exists(base_dir), f'{dataset_name} folder must exist'
    print(f'loading {dataset_name} from {base_dir}')

    csv_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.csv')]
    if dataset_name == 'unraveled':
        for directory in os.listdir(base_dir):
            for filename in os.listdir(os.path.join(base_dir, directory)):
                csv_files.append(os.path.join(base_dir, directory, filename))

    assert len(csv_files) > 0, f'{dataset_name} must contain files (csv)'

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        print(f'loaded {df.shape} from {csv_file}')
        print('label distribution')
        print(df['Stage'].value_counts())
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    return df


def clearn_data(df: pd.DataFrame, impute_nan=False) -> tuple:
    print(f'input shape before = {df.shape}')

    # unify target
    if 'stage' in df.columns:
        df['label'] = df['stage']
        df = df.drop(['stage', 'activity'], axis='columns')
    df.label = df.label.str.lower().replace('normal', 'benign')
    df.label = df.label.str.lower().replace('normaltraffic', 'benign')

    # dropping non-informative / context features
    to_drop = ['source.name', 'flow_id', 'id', 'src_ip', 'dst_ip', 'source_ip', 'destination_ip', 'timestamp',
               'src_port', 'dst_port', 'source_port', 'destination_port', 'protocol', 'src_ip', 'src_mac',
               'src_oui', 'dst_ip', 'dst_mac', 'dst_oui', 'application_name', 'application_category_name',
               'requested_server_name', 'client_fingerprint', 'server_fingerprint',
               'user_agent', 'content_type', 'Unnamed: 0', 'id', 'expiration_id', 'src_port', 'dst_port',
               'protocol', 'ip_version', 'vlan_id', 'tunnel_id', 'bidirectional_first_seen_ms',
               'bidirectional_last_seen_ms',
               'src2dst_first_seen_ms', 'src2dst_last_seen_ms',
               'dst2src_first_seen_ms', 'dst2src_last_seen_ms', 'application_is_guessed']
    drop_cols = [col for col in to_drop if col in df.columns]
    if drop_cols:
        df = df.drop(drop_cols, axis='columns')

    # replace inf value with nan & drop complete nan row/cols
    df = df.replace([np.inf, -np.inf], np.nan)

    nan_vals_count = df.isna().sum().sum()
    print(f'Total number of nan values = {nan_vals_count}')
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)

    if impute_nan:
        if nan_vals_count > 0:
            for col in df.select_dtypes(exclude='object').columns:
                if df[col].isna().sum() > 0:
                    print(f'col {col} has {df[col].isna().sum()} nan values')
                    df[col] = df[col].fillna(df[col].median())

    # remove column with 0 variance
    var = df.var(numeric_only=True)
    cols = var[var == 0].index.values
    df = df.drop(cols, axis='columns')

    print(f'output shape after = {df.shape}')

    return df


def load_train_test_sets(dataset_name='dapt20', anomaly=True, seed=42):
    task = 'anomaly' if anomaly else 'supervised'
    train_set = pd.read_csv(f'./data/{dataset_name}/{task}/train.csv')
    test_set = pd.read_csv(f'./data/{dataset_name}/{task}/test.csv')

    val_set = train_set.sample(frac=0.1, random_state=seed)
    train_set = train_set.drop(val_set.index)

    return train_set, val_set, test_set


def find_net_arch(out_dim, num_hidden, factor=1.0):
    curr = out_dim
    archi = [curr]
    for i in range(num_hidden):
        curr = int(factor * curr)
        archi.append(curr)

    archi.reverse()

    return archi


def tsne_scatter_anomaly(features, labels, dimensions=2, save_as='graph.png', notebook=False):
    sns.set(style='white', context='notebook')

    print(f'{features.shape} , {labels.shape}')
    print(f'{labels.value_counts()}')

    if dimensions not in (2, 3):
        raise ValueError(
            'tsne_scatter can only plot in 2d or 3d (What are you? An alien that can visualise >3d?). Make sure the "dimensions" argument is in (2, 3)')

    features_embedded = TSNE(
        n_components=dimensions,
        perplexity=30,
        random_state=42,
        n_jobs=8
    ).fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 8))

    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    colors = sns.color_palette("tab10", 2)

    ax.scatter(
        *zip(*features_embedded[np.where(labels == 1)]),
        marker='o',
        color=colors[0],
        s=2,
        alpha=0.7,
        label='Attack'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 0)]),
        marker='o',
        color=colors[1],
        s=2,
        alpha=0.3,
        label='Benign'
    )

    plt.legend(loc='best')

    plt.savefig(save_as, bbox_inches='tight', dpi=300, format='svg')

    if notebook:
        plt.show()

    return fig


def tsne_scatter(features, labels, dimensions=2, save_as='graph.png', notebook=False):
    sns.set(style='white', context='notebook')

    print(f'{features.shape} , {labels.shape}')
    unique_labels = np.unique(labels)
    print(f'Label distribution: {dict(zip(unique_labels, [np.sum(labels == l) for l in unique_labels]))}')

    if dimensions not in (2, 3):
        raise ValueError('tsne_scatter can only plot in 2d or 3d. Ensure the "dimensions" argument is either 2 or 3.')

    features_embedded = TSNE(
        n_components=dimensions,
        perplexity=30,
        random_state=42,
        n_jobs=8
    ).fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 8))

    if dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')

    colors = sns.color_palette("tab10", len(unique_labels))

    for label, color in zip(unique_labels, colors):
        indices = np.where(labels == label)
        ax.scatter(
            *zip(*features_embedded[indices]),
            marker='o',
            color=color,
            s=5,
            alpha=0.7,
            label=label
        )

    plt.legend(loc='best', title="Stage")
    plt.savefig(save_as)

    if notebook:
        plt.show()

    return fig

def plot_precision_recall_curve(recall, precision, avg_precision, save_as='figure.svg', notebook=False):
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True)

    figs_dir = f'./figs/{datetime.now().strftime("%Y-%m-%d")}/pr_curves'
    os.makedirs(figs_dir, exist_ok=True)

    plt.savefig(f'{figs_dir}/{save_as}', bbox_inches='tight', dpi=300, format='svg')

    if notebook:
        plt.show()

def plot_roc_curve(fpr, tpr, auc, save_as='figure.svg', notebook=False):
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC-ROC = {auc:.3f}', linewidth=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.grid(True)

    figs_dir = f'./figs/{datetime.now().strftime("%Y-%m-%d")}/roc_curves'
    os.makedirs(figs_dir, exist_ok=True)

    plt.savefig(f'{figs_dir}/{save_as}', bbox_inches='tight', dpi=300, format='svg')

    if notebook:
        plt.show()

