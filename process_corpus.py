#!/usr/bin/env python3
import collections
import glob
import os
import pickle

from deeppavlov import build_model, configs
from razdel import sentenize


def get_unique_out_path(path):
    if not os.path.exists(path):
        return path
    basename, ext = os.path.splitext()
    i = 1
    while True:
        path = f'{basename}_{i}{ext}'
        if not os.path.exists(path):
            return path
        i += 1


SentenceInfo = collections.namedtuple('SentenceInfo',
                                      'span text pos syntax srl'.split(' '))


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    pos_model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy, download=True)
    syntax_model = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)

    for in_path in glob.glob(args.inglob, recursive=True):
        print(in_path)

        docname = os.path.splitext(os.path.basename(in_path))[0]
        out_path = get_unique_out_path(os.path.join(args.outdir, docname + '.pickle'))

        with open(in_path, 'r') as f:
            full_text = f.read()

        sentences_spans = sentenize(full_text)
        sentences_pos = pos_model.batched_call([s.text for s in sentences_spans], batch_size=1)
        sentences_syntax = syntax_model.batched_call([s.text for s in sentences_spans], batch_size=1)
        doc_sentences = [SentenceInfo(span=(span.start, span.stop),
                                      text=span.text,
                                      pos=pos,
                                      syntax=synt,
                                      srl=None)
                         for span, pos, synt in zip(sentences_spans, sentences_pos, sentences_syntax)]
        with open(out_path, 'wb') as f:
            pickle.dump(doc_sentences, f)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('inglob', type=str)
    aparser.add_argument('outdir', type=str)

    main(aparser.parse_args())
