#!/usr/bin/env python3


import os

import numpy as np
import pandas as pd

from narratex.base import pickle_obj, load_all_docs_lazy, load_yaml
from narratex.clustering import build_simple_event_vocab, extract_collocations_count, calc_pmi, select_pairs_by_weights, \
    build_event_vocab_group_by_w2v, get_group2name_by_freq, measure_similarity_by_mutual_features
from narratex.extraction import get_all_events
from narratex.logger import setup_logger
from narratex.segmentation import infer_segmentation, print_topic_model


def main(args):
    config = load_yaml(args.config)

    os.makedirs(args.outdir, exist_ok=True)
    logger = setup_logger(os.path.join(args.outdir, 'extract_raw_events.log'))

    logger.info('Collect events')
    all_events, _ = get_all_events(load_all_docs_lazy(args.indir))
    logger.info(f'Collected {len(all_events)} events')

    logger.info('Build vocab')
    if config.vocab.kind == 'simple':
        group2event, event2group = build_simple_event_vocab(all_events,
                                                            **config.vocab.kwargs)
    elif config.vocab.kind == 'group_by_word2vec':
        group2event, event2group = build_event_vocab_group_by_w2v(all_events,
                                                                  **config.vocab.kwargs)
    elif config.vocab.kind == 'group_by_topics':
        topic_model = infer_segmentation(load_all_docs_lazy(args.indir), **config.vocab.kwargs)
        print_topic_model(os.path.join(args.outdir, 'topics.txt'), topic_model)
    else:
        raise ValueError(f'Unsupported config.vocab.kind: "{config.vocab.kind}"')

    pickle_obj(group2event, os.path.join(args.outdir, 'group2event.pickle'))
    logger.info(f'Grouped events into {len(group2event)} groups')

    logger.info('Print groups to csv')
    group2name = get_group2name_by_freq(group2event)
    group_freq = pd.Series({group2name[g]: len(evs) for g, evs in group2event.items()})
    group_freq.sort_values(ascending=False, inplace=True)
    group_freq.to_csv(os.path.join(args.outdir, 'all_event_groups.csv'), sep='\t')

    logger.info('Find collocations')
    pair_proba, single_proba = extract_collocations_count(load_all_docs_lazy(args.indir),
                                                          event2group,
                                                          min_sent_distance=config.collocations.min_sent_dist,
                                                          max_sent_distance=config.collocations.max_sent_dist)
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
    all_colloc.to_csv(os.path.join(args.outdir, 'all_colloc_pmi.csv'), sep='\t')

    logger.info('Find collocations via joint PMI features')
    mutual_sim = measure_similarity_by_mutual_features(np.clip(pmi, 0, None))
    np.save(os.path.join(args.outdir, 'pmi_cosine_sim.npy'), mutual_sim)

    all_colloc_sim = select_pairs_by_weights(mutual_sim, name_map=group2name)
    all_colloc_sim['first_count'] = group_freq[all_colloc_sim['first']].reset_index(drop=True)
    all_colloc_sim['second_count'] = group_freq[all_colloc_sim['second']].reset_index(drop=True)
    all_colloc_sim.to_csv(os.path.join(args.outdir, 'all_colloc_via_pmi_sim.csv'), sep='\t')


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('indir', type=str, help='Path to corpus')
    aparser.add_argument('outdir', type=str, help='Where to store results')
    aparser.add_argument('config', type=str, help='Config')

    main(aparser.parse_args())
