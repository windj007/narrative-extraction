import collections
import numpy as np
import pymorphy2
import scipy.sparse
import pandas as pd
import scipy.optimize
from russian_tagsets import converters
from gensim.models import KeyedVectors
import annoy

from narratex.logger import LOGGER


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
    def __init__(self, gensim_emb, texts, trees_n=10):
        self.gensim_emb = gensim_emb
        self.morph = pymorphy2.MorphAnalyzer()
        self.tag_conv = converters.converter('opencorpora-int', 'ud20')
        self.tag_cache = {}

        self.id2text = list(sorted(set(texts)))

        textid2tokens = [[tok + '_' + self.get_tag(tok) for tok in txt.split(' ')]
                         for txt in self.id2text]
        tokenid2token = [tok for tok in sorted(set(tok for txt_toks in textid2tokens for tok in txt_toks))
                         if tok in self.gensim_emb.vocab]
        token2tokenid = {tok: i for i, tok in enumerate(tokenid2token)}
        self.tokenid2vec = [self.gensim_emb[tok] for tok in tokenid2token]

        self.tokenid2textid = collections.defaultdict(set)
        self.text2tokenid = collections.defaultdict(set)
        for txt_i, txt_toks in enumerate(textid2tokens):
            txt = self.id2text[txt_i]
            for tok in txt_toks:
                tok_id = token2tokenid.get(tok, None)
                if tok_id is not None:
                    self.tokenid2textid[tok_id].add(txt_i)
                    self.text2tokenid[txt].add(tok_id)

        self.vector_idx = annoy.AnnoyIndex(self.gensim_emb.vectors.shape[1], 'angular')
        for tok_i, tok_vec in enumerate(self.tokenid2vec):
            self.vector_idx.add_item(tok_i, tok_vec)
        self.vector_idx.build(trees_n)

    def find_most_similar(self, query_txt, candidates_n=10, max_cand_tok_dist=1):
        query_token_ids = self.text2tokenid[query_txt]
        if len(query_token_ids) == 0:
            return []

        candidate_text_ids = set()
        for tokid in query_token_ids:
            candidate_text_ids.update(self.tokenid2textid[tokid])
            sim_tok_ids, sim_tok_sims = self.vector_idx.get_nns_by_item(tokid, candidates_n, include_distances=True)
            if len(sim_tok_ids) > 0:
                for other_tokid, tok_sim in zip(sim_tok_ids, sim_tok_sims):
                    if tok_sim <= max_cand_tok_dist:
                        candidate_text_ids.update(self.tokenid2textid[other_tokid])

        query_feats = self.stack_and_norm([self.tokenid2vec[tok_id] for tok_id in query_token_ids])

        candidate_texts = [self.id2text[i] for i in candidate_text_ids if self.id2text[i] != query_txt]
        sims = []
        for other_txt in candidate_texts:
            other_feats = self.stack_and_norm([self.tokenid2vec[tok_id] for tok_id in self.text2tokenid[other_txt]])
            cur_sim = self.calc_sim(query_feats, other_feats)
            sims.append(cur_sim)
        result = sorted(zip(candidate_texts, sims), key=lambda p: p[1], reverse=True)
        return result

    def measure_similarity(self, txt1, txt2):
        txt1_tokens = self.prepare_tokens(txt1)
        txt2_tokens = self.prepare_tokens(txt2)

        if len(txt1_tokens) == 0 or len(txt2_tokens) == 0:
            return 0

        txt1_embs = self.get_embeddings(txt1_tokens)
        txt2_embs = self.get_embeddings(txt2_tokens)

        return self.calc_sim(txt1_embs, txt2_embs)

    def calc_sim(self, txt1_embs, txt2_embs):
        sims = txt1_embs @ txt2_embs.T

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(sims)
        best_sims = sims[row_ind, col_ind]
        sim = best_sims.mean()
        return sim

    def prepare_tokens(self, txt):
        return [tok + '_' + self.get_tag(tok) for tok in txt.split(' ')]

    def get_tag(self, tok):
        cached_tag = self.tag_cache.get(tok, None)
        if cached_tag is not None:
            return cached_tag
        oc_tag = self.morph.parse(tok)[0].tag.POS
        if oc_tag is None:
            LOGGER.warning(f'Could not find POS-tag for token "{tok}": {oc_tag}')
            tag = 'NOTAG'
        else:
            tag = self.tag_conv(oc_tag).split(' ')[0]
        self.tag_cache[tok] = tag
        return tag

    def get_embeddings(self, tokens):
        vectors = [self.gensim_emb[tok] for tok in tokens if tok in self.gensim_emb.vocab]
        if len(vectors) == 0:
            return None
        return self.stack_and_norm(vectors)

    def stack_and_norm(self, vectors):
        result = np.stack(vectors, axis=0)
        result /= np.linalg.norm(result, axis=1, keepdims=True)
        return result


def build_event_vocab_group_by_w2v(all_events, model_path, min_mentions_per_group=10, same_group_threshold=0.6,
                                   warning_group_threshold=0.4, show_progress_freq=100):
    emb = KeyedVectors.load_word2vec_format(model_path, binary=True)
    sim_index = EmbeddingMatchSimilarity(emb,
                                         (ev.features.text for ev in all_events))

    text2group = {}
    event2group = {}
    group2event = {}
    group_n = 0

    for ev_i, event in enumerate(all_events):
        if ev_i % show_progress_freq == 0:
            LOGGER.info(f'Handled {ev_i}/{len(all_events)} events, total groups {len(group2event)}, '
                        f'unique texts {len(text2group)}')

        cur_txt = event.features.text

        if cur_txt in text2group:
            event2group[event.id] = text2group[cur_txt]
        else:
            sim_texts = sim_index.find_most_similar(cur_txt, max_cand_tok_dist=1 - warning_group_threshold)

            best_group = None
            best_sim = 0
            best_match_txt = None
            for other_txt, best_sim in sim_texts:  # sim_texts are sorted by sim descending
                best_group = text2group.get(other_txt, None)
                if best_group is not None:
                    best_match_txt = other_txt
                    break

            if best_group is not None and best_sim >= same_group_threshold:
                LOGGER.info(f'Merge "{cur_txt}" and "{best_match_txt}" into group {best_group}, '
                            f'similarity {best_sim:.2f}')
                event2group[event.id] = best_group
                group2event[best_group].append(event)
                text2group[cur_txt] = best_group
            else:
                if best_group is not None and warning_group_threshold <= best_sim < same_group_threshold:
                    LOGGER.info(f'Did not merge similar "{cur_txt}" and "{best_match_txt}", '
                                f'but not enough, sim {best_sim:.2f}')
                event2group[event.id] = group_n
                text2group[cur_txt] = group_n
                group2event[group_n] = [event]
                group_n += 1

    group2event = {grid: events for grid, events in group2event.items() if len(events) >= min_mentions_per_group}
    group_remap = {grid: i for i, grid in enumerate(sorted(group2event.keys()))}
    group2event = {group_remap[grid]: events for grid, events in group2event.items()}
    event2group = {ev.id: grid for grid, events in group2event.items() for ev in events}

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


def get_group2name_by_freq(group2event):
    result = {}
    for group_id, events in group2event.items():
        texts_by_freq = collections.Counter(ev.features.text for ev in events)
        name = ', '.join(t for t, _ in texts_by_freq.most_common())
        result[group_id] = name
    return result
