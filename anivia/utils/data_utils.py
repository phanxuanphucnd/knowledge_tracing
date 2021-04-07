import gc
import os
import pandas as pd 

from pandas import DataFrame


def load_pickle(data_path):
    """Load data from a pickle file

    :param data_path: Path to the data file (.pkl)
    """
    data_df = pd.read_pickle(data_path)

    try:
        data_df = data_df[data_df['content_type_id'] == False]
    except:
        print("Column `content_type_id` not exists !")
    
    # arrange by timestamp
    data_df = data_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)

    del data_df["timestamp"]
    del data_df["content_type_id"]

    return data_df

def group_by_user_id(data_df: DataFrame):
    """Group a DataFrame by user_id

    :param data_df: A DataFrame
    """
    group = (
        data_df[["user_id", "content_id", "answered_correctly"]]
        .groupby("user_id")
        .apply(lambda r: (r["content_id"].values, r["answered_correctly"].values))
    )

    return group

def train_test_split(data_df: DataFrame, pct: float=0.1):
    """Split DataFrame into Train and Test subsets

    :param data_df: A DataFrame
    :param pct: The ratio to split train/test set
    """
    train_percent = 1 - pct
    train = data_df.iloc[:int(train_percent * len(data_df))]
    test = data_df.iloc[int(train_percent * len(data_df)):]
    print(f"- Shape of Train dataset: {train.shape}")
    print(f"- Shape of Test dataset: {test.shape}")

    train_group = (
        train[['user_id', 'content_id', 'answered_correctly']]
        .groupby('user_id')
        .apply(lambda r: (r['content_id'].values, r['answered_correctly'].values))
    )

    test_group = (
        test[['user_id', 'content_id', 'answered_correctly']]
        .groupby('user_id')
        .apply(lambda r: (r['content_id'].values, r['answered_correctly'].values))
    )

    return train_group, test_group


def get_n_question(data_df: DataFrame):
    """Get the numbers of the skills

    :param data_df: A DataFrame
    """
    skills = data_df["content_id"].unique()
    n_skills = len(skills)
    print(f"\n- Numbers of the questions: {n_skills}")

    return n_skills