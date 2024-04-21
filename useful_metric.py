import argparse
import pandas as pd
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser(description="argument parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--uri", default='99f26e5dc334468ba20cbaa43361a1e9', type=str)
    parser.add_argument("--tidy", default=False, type=bool,required=False)
    parser.add_argument("--index", default='0', type=str)
    parser.add_argument("--env", default='atari_pong', type=str)
    return parser.parse_args()


def plot_results(df_list, env='atari_pong'):
    import numpy as np
    import holoviews as hv
    """
    Given a list of dataframes, plot results on given environment.
    """
    df = pd.concat(df_list)
    # aggregate runs
    df = df.groupby(['method', 'env', 'env_steps'])['return'].agg(['mean', 'std', 'count']).reset_index()
    df['std'] = df['std'].fillna(0)
    df = df.rename(columns={'mean': 'return