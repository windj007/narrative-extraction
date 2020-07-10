#!/usr/bin/env python3

import numpy as np

from narratex.base import load_pickle, save_json
from narratex.clustering import get_group2name_by_freq
from narratex.visualization import make_dendrogram_dict


def main(args):
    group2event = load_pickle(args.group2event)
    group2name = get_group2name_by_freq(group2event)
    pair_weights = np.load(args.pairwise_weights)

    # fill -inf
    min_weight = pair_weights[~np.isinf(pair_weights)].min() - 10
    pair_weights[np.isinf(pair_weights)] = min_weight

    # ensure symmetric
    pair_weights = (pair_weights + pair_weights.T) / 2

    dct = make_dendrogram_dict(pair_weights, group2name, method='single')
    save_json(dct, args.outfile)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('group2event')
    aparser.add_argument('pairwise_weights')
    aparser.add_argument('outfile')

    main(aparser.parse_args())
