import collections
import glob
import os
import pickle
import re

import conllu
import yaml
from easydict import EasyDict as edict
from razdel.substring import Substring


def wrap_edict(obj):
    if isinstance(obj, (list, tuple)):
        return [wrap_edict(el) for el in obj]
    elif isinstance(obj, (dict, collections.OrderedDict)):
        return edict(obj)
    else:
        return obj


SPLITTER_RE = re.compile(r'^\s*(oo\s+)?---+$', flags=re.MULTILINE)
MULTI_LINE_BREAK_RE = re.compile('\n\n+')


def clean_text(t):
    return MULTI_LINE_BREAK_RE.sub('\n', SPLITTER_RE.sub('', t))


def split_long_sentence(s, max_len=512, overlap=0.1):
    if len(s.text) <= max_len:
        return [s]
    else:
        print(f'Long sententce ({len(s.text)} > {max_len})! "{s}"')
        return [Substring(start=s.start + i,
                          stop=s.start + min(i+max_len, len(s.text)),
                          text=s.text[i:i+max_len])
                for i in range(0, len(s.text), max(1, int(max_len * (1 - overlap))))]


def get_unique_out_path(path, force=False):
    if force:
        return path
    if not os.path.exists(path):
        return path
    basename, ext = os.path.splitext(path)
    i = 1
    while True:
        path = f'{basename}_{i}{ext}'
        if not os.path.exists(path):
            return path
        i += 1


def load_doc(fname):
    with open(fname, 'rb') as f:
        doc = wrap_edict(pickle.load(f))
    for sent in doc:
        sent.joint = load_conll_joint(sent)
    return doc


def load_all_docs(dirname):
    return [load_doc(fname) for fname in glob.glob(os.path.join(dirname, '*.pickle'))]


def load_all_docs_lazy(dirname):
    return (load_doc(fname) for fname in glob.glob(os.path.join(dirname, '*.pickle')))


def load_conll_joint(sent_info):
    pos = wrap_edict(conllu.parse(sent_info.pos)[0])
    synt = wrap_edict(conllu.parse(sent_info.syntax)[0])
    assert len(pos) == len(synt)
    for token_pos_dict, token_synt_dict in zip(pos, synt):
        token_pos_dict.head = token_synt_dict.head
        token_pos_dict.deprel = token_synt_dict.deprel
    return pos


def calc_corpus_stat(docs):
    return dict(total_docs=len(docs),
                total_sentences=sum(len(doc) for doc in docs),
                total_tokens=sum(len(sent.joint) for doc in docs for sent in doc))


def pickle_obj(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_yaml(fname):
    with open(fname, 'r') as f:
        return edict(yaml.safe_load(f))
