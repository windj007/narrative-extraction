#!/usr/bin/env python3

import os

import numpy as np

from narratex.base import load_pickle, load_all_docs_lazy, load_yaml, pickle_obj
from narratex.clustering import extract_assoc_rules, get_group2name_by_freq


def main(args):
    config = load_yaml(args.config)
    os.makedirs(args.outdir)

    group2event = load_pickle(os.path.join(args.stats_indir, 'group2event.pickle'))
    event2group = {ev.id: gr for gr, events in group2event.items() for ev in events}
    group2name = get_group2name_by_freq(group2event)

    pmi = np.load(os.path.join(args.stats_indir, 'pmi.npy'))
    single_proba = np.load(os.path.join(args.stats_indir, 'single_proba.npy'))

    weighted_rules = extract_assoc_rules(load_all_docs_lazy(args.docs_indir), single_proba, pmi, event2group,
                                         **config.assoc_kwargs)
    pickle_obj(weighted_rules, os.path.join(args.outdir, 'weighted_rules.npy'))

    with open(os.path.join(args.outdir, 'weighted_rules.csv'), 'w') as outf:
        for weight, itemset in weighted_rules:
            title = '\t'.join(group2name[gr] for gr in itemset)
            outf.write(f'{weight:.3f}\t{title}\n')


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str)
    aparser.add_argument('docs_indir', type=str)
    aparser.add_argument('stats_indir', type=str)
    aparser.add_argument('outdir', type=str)

    main(aparser.parse_args())
