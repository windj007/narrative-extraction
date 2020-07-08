import collections
import numpy as np
import pymorphy2
import scipy.sparse
import pandas as pd
import scipy.optimize
from russian_tagsets import converters
from gensim.models import KeyedVectors


def build_simple_event_vocab(all_events, min_mentions_per_group=10):
    events_by_id = {ev.id: ev for ev in all_events}
    text2events = collections.defaultdict(list)
    for event in all_events:
        text2events[event.features.text].append(event.id)
    text2events = {k: v for k, v in text2events.items() if len(v) >= min_mentions_per_group}

    key2group = {key: i for i, key in enumerate(sorted(text2events.keys()))}
    event2group = {evid: key2group[key]
                   for key, group_events in text2events.items()
                   for evid in group_events}
    group2event = {key2group[key]: [events_by_id[i] for i in group_events]
                   for key, group_events in text2events.items()}
    return group2event, event2group


class EmbeddingMatchSimilarity:
    def __init__(self, gensim_emb):
        self.gensim_emb = gensim_emb
        self.morph = pymorphy2.MorphAnalyzer()
        self.tag_conv = converters.converter('opencorpora-int', 'ud20')

    def __call__(self, txt1, txt2):
        print('txt1', txt1)
        print('txt2', txt2)
        txt1_tokens = self.prepare_tokens(txt1)
        txt2_tokens = self.prepare_tokens(txt2)
        print('txt1_tokens', txt1_tokens)
        print('txt2_tokens', txt2_tokens)

        if len(txt1_tokens) == 0 or len(txt2_tokens) == 0:
            return 0

        txt1_embs = self.get_embeddings(txt1_tokens)
        txt2_embs = self.get_embeddings(txt2_tokens)

        print('txt1_embs', txt1_embs.shape)
        print('txt2_embs', txt2_embs.shape)

        sims = txt1_embs @ txt2_embs.T

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(sims)
        best_sims = sims[row_ind, col_ind]
        return best_sims.mean()

    def prepare_tokens(self, txt):
        return [tok + '_' + self.get_tag(tok) for tok in txt.split(' ')]

    def get_tag(self, tok):
        oc_tag = self.morph.parse(tok)[0].tag.POS
        return self.tag_conv(oc_tag).split(' ')[0]

    def get_embeddings(self, tokens):
        result = np.stack([self.gensim_emb[tok] for tok in tokens if tok in self.gensim_emb.vocab], axis=0)
        result /= np.linalg.norm(result, axis=1, keepdims=True)
        return result


def build_event_vocab_group_by_w2v(all_events, model_path, min_mentions_per_group=10, same_group_threshold=0.6):
    emb = KeyedVectors.load_word2vec_format(model_path, binary=True)
    measure = EmbeddingMatchSimilarity(emb)
    text2group = {}
    event2group = {}
    group_n = 0

    for event in all_events:
        cur_txt = event.features.text

        if cur_txt in text2group:
            event2group[event.id] = text2group[cur_txt]
        else:
            best_group = None
            best_sim = 0

            for other_txt, other_group_id in text2group.items():
                cur_sim = measure(cur_txt, other_txt)
                if cur_sim >= best_sim or best_group is None:
                    best_sim = cur_sim
                    best_group = other_group_id

            if best_group is not None and best_sim >= same_group_threshold:
                event2group[event.id] = best_group
                text2group[cur_txt] = best_group
            else:
                event2group[event.id] = group_n
                text2group[cur_txt] = group_n
                group_n += 1

    group2event = {}
    for evid, grid in event2group.items():
        if grid in group2event:
            group2event[grid].append(evid)
        else:
            group2event[grid] = [evid]

    for grid, group_evs in list(group2event):
        if len(group_evs) < min_mentions_per_group:
            del group2event[grid]
            for evid in group_evs:
                del event2group[evid]

    return group2event, event2group


def extract_collocations_count(docs, event2group, min_sent_distance=0, max_sent_distance=3):
    assert min_sent_distance >= 0
    assert min_sent_distance <= max_sent_distance
    n_groups = max(event2group.values()) + 1
    pair_counts = scipy.sparse.dok_matrix((n_groups, n_groups))
    event_counts = np.zeros(n_groups)
    sent_number = 0

    for doc in docs:
        for sent1_i, sent1 in enumerate(doc):
            sent_number += 1

            left_i = max(0, sent1_i - max_sent_distance)
            for ev1 in sent1.events:
                if ev1.id in event2group:
                    event_counts[event2group[ev1.id]] += 1

            for sent2_i in range(left_i, sent1_i + 1 - min_sent_distance):
                sent2 = doc[sent2_i]
                for ev1 in sent1.events:
                    if ev1.id not in event2group:
                        continue
                    eg1 = event2group[ev1.id]
                    for ev2 in sent2.events:
                        if ev1.id == ev2.id or ev2.id not in event2group:
                            continue
                        eg2 = event2group[ev2.id]
                        pair_counts[eg1, eg2] += 1
                        pair_counts[eg2, eg1] += 1

    pair_proba = pair_counts.toarray() / sent_number
    single_proba = event_counts / sent_number
    return pair_proba, single_proba


def calc_pmi(pair_proba, single_proba):
    norm = single_proba[None, ...] * single_proba[..., None] + 1e-8
    return np.log(pair_proba / norm)


def select_pairs_by_weights(pairwise_weights, name_map=None, min_weight=0):
    first_index, second_index = np.where(pairwise_weights > min_weight)
    pairs = list({tuple(sorted((a, b))) for a, b in zip(first_index, second_index) if a != b})
    first_index, second_index = zip(*pairs)
    weights = pairwise_weights[first_index, second_index].reshape(-1)
    if name_map is not None:
        first_index = [name_map[i] for i in first_index]
        second_index = [name_map[i] for i in second_index]
    result = pd.DataFrame(dict(first=first_index, second=second_index, pmi=weights))
    result.sort_values('pmi', ascending=False, inplace=True)
    result.reset_index(inplace=True, drop=True)
    return result
