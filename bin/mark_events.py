#!/usr/bin/env python3


import glob
import os
from functools import partial

from joblib import Parallel, delayed

from narratex.base import load_doc, pickle_obj
from narratex.extraction import find_predicates_simple, simple_event_features, EventExtractor, mark_events_corpus


def get_out_fname(in_doc_fname, out_corpus_dir):
    return os.path.join(out_corpus_dir, os.path.basename(in_doc_fname))


def mart_events_one(i, in_doc_fname, out_corpus_dir, evex):
    if i % 10 == 0:
        print(i)

    doc = load_doc(in_doc_fname)
    mark_events_corpus((doc,), evex)
    pickle_obj(doc, get_out_fname(in_doc_fname, out_corpus_dir))


def main(args):
    out_corpus_dir = os.path.join(args.outdir, 'docs_with_events')
    os.makedirs(out_corpus_dir, exist_ok=True)

    evex = EventExtractor(partial(find_predicates_simple,
                                  obj_max_depth=2,
                                  verbal_nouns_mode=None),
                          simple_event_features)

    Parallel(n_jobs=args.jobs_n)(delayed(mart_events_one)(i, fname, out_corpus_dir, evex)
                                 for i, fname in enumerate(glob.glob(os.path.join(args.indir, '*.pickle')))
                                 if not os.path.exists(get_out_fname(fname, out_corpus_dir)))


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('indir', type=str, help='Path to corpus')
    aparser.add_argument('outdir', type=str, help='Where to store results')
    aparser.add_argument('--jobs-n', type=int, default=-1,
                         help='Number of processes')

    main(aparser.parse_args())
