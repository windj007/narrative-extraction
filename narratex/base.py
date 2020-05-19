import collections
import os
import pickle
import re

import brave
import conllu
from easydict import EasyDict as edict


def wrap_edict(obj):
    if isinstance(obj, (list, tuple)):
        return [wrap_edict(el) for el in obj]
    elif isinstance(obj, (dict, collections.OrderedDict)):
        return edict(obj)
    else:
        return obj


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


def load_conll_joint(sent_info):
    pos = wrap_edict(conllu.parse(sent_info.pos)[0])
    synt = wrap_edict(conllu.parse(sent_info.syntax)[0])
    assert len(pos) == len(synt)
    for token_pos_dict, token_synt_dict in zip(pos, synt):
        token_pos_dict.head = token_synt_dict.head
        token_pos_dict.deprel = token_synt_dict.deprel
    return pos


DETOKENIZE_RULES = (
    (r' ([.,!?;:»%)\]\'])( ?)', r'\1\2'),
    (r'( ?)([([«`~@]) ', r'\1\2')
)
DETOKENIZE_RULES = [(re.compile(rule), rep) for rule, rep in DETOKENIZE_RULES]


def infer_spans(text, tokens, unk='[UNK]', unk_sep=' '):
    token_spans = []
    cur_pos = 0
    for i, tok in enumerate(tokens):
        if tok == unk:
            if i < len(tokens) - 1:
                next_not_unk = i + 1
                while next_not_unk < len(tokens) and tokens[next_not_unk] == unk:
                    next_not_unk += 1

                if next_not_unk > i + 1:
                    if next_not_unk == len(tokens):
                        next_not_unk_start = len(text)
                    else:
                        next_not_unk_start = text.find(tokens[next_not_unk], cur_pos)

                    next_start = text.find(unk_sep, cur_pos + 1, next_not_unk_start)
                    if next_start == -1:
                        next_start = next_not_unk_start
                else:
                    next_start = text.find(tokens[next_not_unk], cur_pos)

                assert next_start >= cur_pos, (text, tokens, i, tok)
                tok = text[cur_pos:next_start].strip()
            else:
                tok = text[cur_pos:].strip()

        new_pos = text.find(tok, cur_pos)
        assert new_pos >= cur_pos, (text, tokens, tok, cur_pos, new_pos)
        token_spans.append((new_pos, new_pos + len(tok)))
        cur_pos = new_pos + len(tok)
    assert len(tokens) == len(token_spans)
    return token_spans


def detokenize_sentence(tokens):
    result = ' '.join(tokens)
    while True:
        new_result = result
        for regex, rep in DETOKENIZE_RULES:
            new_result = regex.sub(rep, new_result)
        if new_result == result:
            break
        result = new_result
    return result, infer_spans(result, tokens)


def joint_markup_to_brat(tokens, syntax=True):
    text, token_spans = detokenize_sentence([t.form for t in tokens])
    entities = []
    attributes = []
    for tok in tokens:
        entities.append((f'token_{tok.id}',
                         tok.upostag,
                         [token_spans[tok.id - 1]]))
    relations = []
    if syntax:
        for tok in tokens:
            if tok.head is not None and tok.head > 0:
                relations.append((f'dep_{len(relations)}',
                                  f'synt-{tok.deprel}',
                                  (('child', f'token_{tok.id}'),
                                   ('head', f'token_{tok.head}'))))
    return dict(text=text,
                entities=entities,
                relations=relations,
                attributes=attributes)


def brave_visualize_sent(sent, **kwargs):
    return brave.brave(joint_markup_to_brat(sent.joint, **kwargs),
                       dict())
