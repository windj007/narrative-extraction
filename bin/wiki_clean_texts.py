#!/usr/bin/env python3


import re
import glob
import os


HEADER_RE = re.compile(r'^\s*=+\s*(.+?)\s*=+$', re.M)
DROP_RE = re.compile(r"(''+|^(Файл|Категория):.+?$)", re.M)


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    for in_fname in glob.glob(args.inglob):
        with open(in_fname, 'r') as f:
            text = f.read()
        text = DROP_RE.sub('', text)
        text = HEADER_RE.sub(r'\n\1.\n', text)

        with open(os.path.join(args.outdir, os.path.basename(in_fname)), 'w') as f:
            f.write(text)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('inglob', type=str, help='Pattern to get paths to text files')
    aparser.add_argument('outdir', type=str, help='Where to store results')

    main(aparser.parse_args())
