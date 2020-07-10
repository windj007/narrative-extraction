#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

from narratex.base import load_pickle
from narratex.clustering import select_pairs_by_weights, get_group2name_by_freq


def bin_entropy(p, eps=1e-10):
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def information_gain(fore_p, back_p):
    fore_ent = bin_entropy(fore_p)
    back_ent = bin_entropy(back_p)
    return back_ent - fore_ent


LOG_EPS = 1e-5


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    back_group2event = load_pickle(os.path.join(args.background, 'group2event.pickle'))
    back_pair_proba = np.load(os.path.join(args.background, 'pair_proba.npy'))
    back_single_proba = np.load(os.path.join(args.background, 'single_proba.npy'))

    fore_group2event = load_pickle(os.path.join(args.foreground, 'group2event.pickle'))
    fore_pair_proba = np.load(os.path.join(args.foreground, 'pair_proba.npy'))
    fore_single_proba = np.load(os.path.join(args.foreground, 'single_proba.npy'))

    # remap background probabilities to the shape of foreground ones
    back_name2group = {back_subname: gr
                       for gr, back_name in get_group2name_by_freq(back_group2event).items()
                       for back_subname in back_name.split(', ')}
    fore_group2name = get_group2name_by_freq(fore_group2event)
    fore2back_group_map = {fore_gr: {back_name2group[fore_subname]
                                     for fore_subname in fore_name.split(', ')
                                     if fore_subname in back_name2group}
                           for fore_gr, fore_name in fore_group2name.items()}
    back2fore_mapped_single_proba = np.array([sum(back_single_proba[back_gr]
                                                  for back_gr in fore2back_group_map[fore_gr])
                                              for fore_gr in range(len(fore_single_proba))])
    # back2fore_mapped_pair_proba = np.array([[sum(back_pair_proba[back_gr1, back_gr2]
    #                                              for back_gr1 in fore2back_group_map[fore_gr1]
    #                                              for back_gr2 in fore2back_group_map[fore_gr2])
    #                                          for fore_gr2 in range(len(fore_single_proba))]
    #                                         for fore_gr1 in range(len(fore_single_proba))])

    fore_single_count = pd.Series({fore_group2name[gr]: len(evs) for gr, evs in fore_group2event.items()})
    back2fore_single_count = pd.Series({fore_group2name[gr]: sum(len(back_group2event[back_gr])
                                                                 for back_gr in fore2back_group_map[gr])
                                        for gr in fore_group2event.keys()})

    # contrastive reweighting based only on single probability difference
    single_logtfidf = np.log(fore_single_proba + LOG_EPS) - np.log(back2fore_mapped_single_proba + LOG_EPS)
    np.save(os.path.join(args.outdir, 'single_proba.npy'), single_logtfidf)
    group_freq_single_contr = pd.Series({name: single_logtfidf[gr] for gr, name in fore_group2name.items()})
    group_freq_single_contr.sort_values(ascending=False, inplace=True)
    group_freq_single_contr.to_csv(os.path.join(args.outdir, 'all_event_groups.csv'), sep='\t')

    pmi_single_contr = np.log(fore_pair_proba + LOG_EPS) + single_logtfidf[None, ...] + single_logtfidf[..., None]
    np.save(os.path.join(args.outdir, 'pmi.npy'), pmi_single_contr)

    pmi_threshold1 = np.quantile(pmi_single_contr.reshape(-1), 0.8)

    all_colloc_single_contr = select_pairs_by_weights(pmi_single_contr,
                                                      name_map=fore_group2name,
                                                      min_weight=pmi_threshold1)
    all_colloc_single_contr['first_foreground_count'] = fore_single_count[all_colloc_single_contr['first']] \
        .reset_index(drop=True)
    all_colloc_single_contr['second_foreground_count'] = fore_single_count[all_colloc_single_contr['second']] \
        .reset_index(drop=True)
    all_colloc_single_contr['first_background_count'] = back2fore_single_count[all_colloc_single_contr['first']] \
        .reset_index(drop=True)
    all_colloc_single_contr['second_background_count'] = back2fore_single_count[all_colloc_single_contr['second']] \
        .reset_index(drop=True)
    all_colloc_single_contr.to_csv(os.path.join(args.outdir, 'all_colloc.csv'), sep='\t')

    # contrastive reweighting based on single and pairwise probability difference
    # pmi_pair_contr = pmi_single_contr - np.log(back2fore_mapped_pair_proba + LOG_EPS)
    # np.save(os.path.join(args.outdir, 'pmi_pair_contrast.npy'), pmi_pair_contr)
    #
    # pmi_threshold2 = np.quantile(pmi_pair_contr.reshape(-1), 0.8)

    # all_colloc_pair_contr = select_pairs_by_weights(pmi_pair_contr,
    #                                                 name_map=fore_group2name,
    #                                                 min_weight=pmi_threshold2)
    # all_colloc_pair_contr['first_foreground_count'] = fore_single_count[all_colloc_pair_contr['first']] \
    #     .reset_index(drop=True)
    # all_colloc_pair_contr['second_foreground_count'] = fore_single_count[all_colloc_pair_contr['second']] \
    #     .reset_index(drop=True)
    # all_colloc_pair_contr['first_background_count'] = back2fore_single_count[all_colloc_pair_contr['first']] \
    #     .reset_index(drop=True)
    # all_colloc_pair_contr['second_background_count'] = back2fore_single_count[all_colloc_pair_contr['second']] \
    #     .reset_index(drop=True)
    # all_colloc_pair_contr.to_csv(os.path.join(args.outdir, 'all_colloc_pair_contr.csv'), sep='\t')


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('background', type=str,
                         help='Path to folder with counters from background collection')
    aparser.add_argument('foreground', type=str,
                         help='Path to folder with counters from the collection to analyze')
    aparser.add_argument('outdir', type=str,
                         help='Where to store the results')

    main(aparser.parse_args())
