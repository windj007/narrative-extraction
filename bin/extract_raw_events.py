#!/usr/bin/env python3


import glob
import os
from functools import partial

import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed

from narratex.base import load_doc, pickle_obj
from narratex.clustering import build_simple_event_vocab, extract_collocations_count, calc_pmi, select_pairs_by_pmi
from narratex.extraction import find_predicates_simple, simple_event_features, EventExtractor, mark_events_corpus, \
    get_all_events
from narratex.logger import setup_logger


def mart_events_one(in_doc_fname, out_corpus_dir, evex):
    doc = load_doc(in_doc_fname)
    mark_events_corpus((doc,), evex)
    pickle_obj(doc, os.path.join(out_corpus_dir, os.path.basename(in_doc_fname)))
    return doc


def main(args):
    out_corpus_dir = os.path.join(args.outdir, 'docs_with_events')
    os.makedirs(out_corpus_dir, exist_ok=True)

    logger = setup_logger(os.path.join(args.outdir, 'extract_raw_events.log'))

    evex = EventExtractor(partial(find_predicates_simple,
                                  obj_max_depth=2,
                                  verbal_nouns_mode=None),
                          simple_event_features)

    logger.info('Mark events')
    corpus = Parallel(n_jobs=args.jobs_n)(delayed(mart_events_one)(fname, out_corpus_dir, evex)
                                          for fname in glob.glob(os.path.join(args.indir, '*.pickle')))
    logger.info(f'Total docs {len(corpus)}')

    logger.info('Collect events')
    all_events, event2ds = get_all_events(corpus)
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
    pair_proba, single_proba = extract_collocations_count(corpus, event2group,
                                                          max_sent_distance=args.max_sent_dist)
    np.save(os.path.join(args.outdir, 'pair_proba.npy'), pair_proba)
    np.save(os.path.join(args.outdir, 'single_proba.npy'), single_proba)
    logger.info('Collocations done')

    logger.info('Calc PMI')
    pmi = calc_pmi(pair_proba, single_proba)
    np.save(os.path.join(args.outdir, 'pmi.npy'), pmi)
    logger.info('PMI done')

    logger.info('Print pairs to csv')
    all_colloc = select_pairs_by_pmi(pmi, name_map=group2name)
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
