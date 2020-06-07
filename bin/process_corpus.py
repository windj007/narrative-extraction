#!/usr/bin/env python3
import glob
import os
import pickle
import traceback

from deeppavlov import build_model, configs
from razdel import sentenize

from narratex.base import split_long_sentence, clean_text


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # pos_model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy, download=True)
    pos_model = build_model(configs.morpho_tagger.BERT.morpho_ru_syntagrus_bert, download=True)
    syntax_model = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=True)

    for in_path in glob.glob(args.inglob, recursive=True):
        try:
            print(in_path)

            docname = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(args.outdir, docname + '.pickle')

            if os.path.exists(out_path) and not args.f:
                print('Already processed')
                continue

            with open(in_path, 'r') as f:
                full_text = clean_text(f.read())

            sentences_spans = list(sentenize(full_text))
            sentences_spans = [split_sent
                               for sent in sentences_spans
                               for split_sent in split_long_sentence(sent, max_len=args.max_sent_len)]
            sentences_texts = [s.text for s in sentences_spans]
            sentences_pos = pos_model.batched_call(sentences_texts, batch_size=args.batch_size)
            sentences_syntax = syntax_model.batched_call(sentences_texts, batch_size=args.batch_size)
            assert len(sentences_spans) == len(sentences_pos) == len(sentences_syntax)

            doc_sentences = [dict(span=(span.start, span.stop),
                                  text=span.text,
                                  pos=pos,
                                  syntax=synt)
                             for span, pos, synt in zip(sentences_spans, sentences_pos, sentences_syntax)]
            with open(out_path, 'wb') as f:
                pickle.dump(doc_sentences, f)
        except Exception as ex:
            print(f'Failed to process {in_path} due to {ex}\n{traceback.format_exc()}')


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('inglob', type=str)
    aparser.add_argument('outdir', type=str)
    aparser.add_argument('-f', action='store_true')
    aparser.add_argument('--max-sent-len', type=int, default=512, help='Maximum length of sentence')
    aparser.add_argument('--batch-size', type=int, default=1, help='Batch size for BERT')

    main(aparser.parse_args())
