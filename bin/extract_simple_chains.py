#!/usr/bin/env python3


import os

import numpy as np
import pandas as pd

from narratex.base import pickle_obj, load_all_docs_lazy
from narratex.clustering import build_simple_event_vocab, extract_collocations_count, calc_pmi, select_pairs_by_weights
from narratex.extraction import get_all_events
from narratex.logger import setup_logger


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    logger = setup_logger(os.path.join(args.outdir, 'extract_raw_events.log'))

    logger.info('Collect events')
    all_events, _ = get_all_events(load_all_docs_lazy(args.indir))
    logger.info(f'Collected {len(all_events)} events')

    logger.info('Build vocab')
    group2event, event2group = build_simple_event_vocab(all_events,
                                                        min_mentions_per_group=args.min_mentions)
    pickle_obj(group2event, os.path.join(args.outdir, 'group2event.pickle'))
    logger.info(f'Grouped events into {len(group2event)} groups')

    logger.info('Print groups to csv')
    group2name = {gr: ev[0].features.text for gr, ev in group2event.items()}
    group_freq = pd.Series({group2name[g]: len(evs) for g, evs in group2event.items()})
    group_freq.sort_values(ascending=False, inplace=True)
    group_freq.to_csv(os.path.join(args.outdir, 'all_event_groups.csv'), sep='\t')

    logger.info('Find collocations')
    pair_proba, single_proba = extract_collocations_count(load_all_docs_lazy(args.indir),
                                                          event2group,
                                                          max_sent_distance=args.max_sent_dist)
    np.save(os.path.join(args.outdir, 'pair_proba.npy'), pair_proba)
    np.save(os.path.join(args.outdir, 'single_proba.npy'), single_proba)
    logger.info('Collocations done')

    logger.info('Calc PMI')
    pmi = calc_pmi(pair_proba, single_proba)
    np.save(os.path.join(args.outdir, 'pmi.npy'), pmi)
    logger.info('PMI done')

    logger.info('Print pairs to csv')
    all_colloc = select_pairs_by_weights(pmi, name_map=group2name)
    all_colloc['first_count'] = group_freq[all_colloc['first']].reset_index(drop=True)
    all_colloc['second_count'] = group_freq[all_colloc['second']].reset_index(drop=True)
    all_colloc.to_csv(os.path.join(args.outdir, 'all_colloc.csv'), sep='\t')


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('indir', type=str, help='Path to corpus')
    aparser.add_argument('outdir', type=str, help='Where to store results')
    aparser.add_argument('--min-mentions', type=int, default=50, help='Minimum mentions number for event to persist')
    aparser.add_argument('--max-sent-dist', type=int, default=3,
                         help='Maximum number of sentences between actions to count co-occurrence')
    aparser.add_argument('--jobs-n', type=int, default=-1,
                         help='Number of processes')

    main(aparser.parse_args())
