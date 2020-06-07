#!/usr/bin/env python3

import bz2
import glob
import os

from gensim.corpora.wikicorpus import extract_pages


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    filter_namespaces = ('0',)

    out_i = 0
    for in_fname in glob.glob(args.inglob):
        for title, text, pageid in extract_pages(bz2.BZ2File(in_fname),
                                                 filter_namespaces=filter_namespaces):
            if out_i % args.skip == 0:
                with open(os.path.join(args.outdir, f'{pageid}.txt'), 'w') as f:
                    f.write(text)
            out_i += 1


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('inglob', type=str, help='Pattern to get wikipedia dumps')
    aparser.add_argument('outdir', type=str, help='Where to store extracted texts')
    aparser.add_argument('--skip', type=int, default=10, help='How many docs to skip between two docs to save')

    main(aparser.parse_args())
