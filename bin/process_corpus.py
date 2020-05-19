#!/usr/bin/env python3
import glob
import os
import pickle

from deeppavlov import build_model, configs
from razdel import sentenize

from narratex.base import get_unique_out_path


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # pos_model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy, download=True)
    pos_model = build_model(configs.morpho_tagger.BERT.morpho_ru_syntagrus_bert, download=True)
    syntax_model = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)

    for in_path in glob.glob(args.inglob, recursive=True):
        print(in_path)

        docname = os.path.splitext(os.path.basename(in_path))[0]
        out_path = get_unique_out_path(os.path.join(args.outdir, docname + '.pickle'), force=args.f)

        with open(in_path, 'r') as f:
            full_text = f.read()

        sentences_spans = list(sentenize(full_text))[:2]
        sentences_pos = pos_model.batched_call([s.text for s in sentences_spans], batch_size=1)
        sentences_syntax = syntax_model.batched_call([s.text for s in sentences_spans], batch_size=1)
        assert len(sentences_spans) == len(sentences_pos) == len(sentences_syntax)

        doc_sentences = [dict(span=(span.start, span.stop),
                              text=span.text,
                              pos=pos,
                              syntax=synt)
                         for span, pos, synt in zip(sentences_spans, sentences_pos, sentences_syntax)]
        with open(out_path, 'wb') as f:
            pickle.dump(doc_sentences, f)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('inglob', type=str)
    aparser.add_argument('outdir', type=str)
    aparser.add_argument('-f', action='store_true')

    main(aparser.parse_args())
