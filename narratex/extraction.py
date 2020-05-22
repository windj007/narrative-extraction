import os
from easydict import EasyDict as edict
import uuid
import re
import toposort


_VERBAL_NOUNS = None


def get_verbal_nouns():
    global _VERBAL_NOUNS
    if _VERBAL_NOUNS is None:
        fname = os.path.join(os.path.dirname(__file__), 'data', 'verbal_nouns.txt')
        with open(fname, 'r') as f:
            _VERBAL_NOUNS = set(l.strip().lower() for l in f)
    return _VERBAL_NOUNS


VERBAL_NOUN_RE = re.compile(r'(ание|ение)$', re.I)


GOOD_DEPS = {'xcomp', 'csubj:pass', 'advmod'}
OBJ_DEPS = {'obj'}


def get_tree_distance(sent, group, cur_token_id, target_postag='VERB'):
    result = 1
    while True:
        cur_token_info = sent[cur_token_id - 1]
        if cur_token_info.head not in group:
            return float('inf')
        head = sent[cur_token_info.head - 1]
        if head.upostag == target_postag:
            return result
        cur_token_id = head.id
        result += 1


def find_predicates_simple(sent, verb_deps=GOOD_DEPS, obj_max_depth=-1, obj_deps=OBJ_DEPS, verbal_nouns_mode=None):
    result = []
    for tok in sent:
        if tok.upostag == 'VERB':
            appended_to_existing = False
            if tok.head > 0 and verb_deps and tok.deprel in verb_deps:
                for group in result:
                    if tok.head in group:
                        group.append(tok.id)
                        appended_to_existing = True
                        break
            if not appended_to_existing:
                result.append([tok.id])

        elif obj_max_depth > 0:
            if tok.head > 0 and tok.deprel in obj_deps:
                for group in result:
                    if tok.head in group:
                        cur_depth = get_tree_distance(sent, group, tok.id)
                        if cur_depth <= obj_max_depth:
                            group.append(tok.id)
                            break

        elif tok.upostag == 'NOUN':
            if verbal_nouns_mode is not None:
                if ((verbal_nouns_mode == 'vocab' and tok.lemma in get_verbal_nouns())
                        or (verbal_nouns_mode == 'regex' and VERBAL_NOUN_RE.search(tok.lemma))):
                    result.append([tok.id])
    return result


def get_text(sent, token_ids):
    token_ids = sorted(token_ids)
    return ' '.join(sent.joint[i - 1].lemma for i in token_ids)


# def normalize_group(sent, token_ids):
#     token_cmp = {t: {sent.joint[t - 1].head}
#                  for t in token_ids}
#


def simple_event_features(sent, event):
    return dict(text=get_text(sent, event.action))


class EventExtractor:
    def __init__(self, predicate_extractor, feature_extractor, min_text_len=4):
        self.predicate_extractor = predicate_extractor
        self.feature_extractor = feature_extractor
        self.min_text_len = min_text_len

    def __call__(self, sent):
        predicates = self.predicate_extractor(sent.joint)
        result = [edict(id=uuid.uuid4().hex,
                        action=pred_tokens)
                  for pred_tokens in predicates]
        for event in result:
            event.features = self.feature_extractor(sent, event)
        result = [ev for ev in result if len(ev.features.text) >= self.min_text_len]
        return result


def mark_events_corpus(docs, group_extractor):
    for doc in docs:
        for sent in doc:
            sent.events = list(group_extractor(sent))


def get_all_events(docs):
    event2doc_sent = {}
    result = []
    for doc_i, doc in enumerate(docs):
        for sent_i, sent in enumerate(doc):
            for event in sent.get('events', []):
                result.append(event)
                event2doc_sent[event.id] = (doc_i, sent_i)
    return result, event2doc_sent
