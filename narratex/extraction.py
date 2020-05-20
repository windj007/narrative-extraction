import os
from easydict import EasyDict as edict
import uuid
import re

_VERBAL_NOUNS = None


def get_verbal_nouns():
    global _VERBAL_NOUNS
    if _VERBAL_NOUNS is None:
        fname = os.path.join(os.path.dirname(__file__), 'data', 'verbal_nouns.txt')
        with open(fname, 'r') as f:
            _VERBAL_NOUNS = set(l.strip().lower() for l in f)
    return _VERBAL_NOUNS


VERBAL_NOUN_RE = re.compile(r'(ание|ение)$', re.I)


def find_predicates_simple(sent, verbal_nouns_mode=None):
    result = []
    for tok in sent:
        if tok.upostag == 'VERB':
            result.append([tok.id])
        elif tok.upostag == 'NOUN':
            if verbal_nouns_mode is not None:
                if ((verbal_nouns_mode == 'vocab' and tok.lemma in get_verbal_nouns())
                        or (verbal_nouns_mode == 'regex' and VERBAL_NOUN_RE.search(tok.lemma))):
                    result.append([tok.id])
    return result


def get_text(sent, token_ids):
    token_ids = sorted(token_ids)
    return ' '.join(sent.joint[i - 1].lemma for i in token_ids)


def simple_event_features(sent, event):
    return dict(text=get_text(sent, event.action))


class EventExtractor:
    def __init__(self, predicate_extractor, feature_extractor):
        self.predicate_extractor = predicate_extractor
        self.feature_extractor = feature_extractor

    def __call__(self, sent):
        predicates = self.predicate_extractor(sent.joint)
        result = [edict(id=uuid.uuid4().hex,
                        action=pred_tokens)
                  for pred_tokens in predicates]
        for event in result:
            event.features = self.feature_extractor(sent, event)
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
