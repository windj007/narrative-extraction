#!/usr/bin/env python3


import os
import re

import pandas as pd

BLACKLIST_FILE = os.path.join(os.path.dirname(__file__), '..', 'narratex', 'data', 'events_blacklist.txt')


def main(args):
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)

    with open(BLACKLIST_FILE, 'r') as f:
        blacklist_patterns = [re.compile(line.strip(), re.I) for line in f if line.strip()]

    if args.is_frame:
        df = pd.read_csv(args.inpath, sep='\t', index_col=[0], header=1)
    else:
        df = pd.read_csv(args.inpath, sep='\t', header=0)

    import pdb; pdb.set_trace()
    str_columns = [c for c in df.columns if df[c].dtype == 'str']
    df_str_cols = df[str_columns]
    save_filter = [not any(pattern.search(val)
                           for val in df_str_cols.loc[i].values
                           for pattern in blacklist_patterns)
                   for i in df_str_cols.index]
    df = df.loc[save_filter]

    if args.is_frame:
        df.to_csv(args.outpath, sep='\t')
    else:
        df.to_csv(args.outpath, sep='\t', index=False)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('inpath', type=str, help='Pattern to csv file to filter')
    aparser.add_argument('outpath', type=str, help='Where to store results')
    aparser.add_argument('--is-frame', action='store_true',
                         help='Treat csv as dataframe (if absent, csv will be treated as series)')

    main(aparser.parse_args())
