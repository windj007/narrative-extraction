import collections
import numpy as np
import scipy.sparse
import pandas as pd


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


def extract_collocations_count(docs, event2group, max_sent_distance=3):
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

            for sent2_i in range(left_i, sent1_i + 1):
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


def select_pairs_by_pmi(pmi, name_map=None, min_pmi=0):
    first_index, second_index = np.where(pmi > min_pmi)
    pairs = list({tuple(sorted((a, b))) for a, b in zip(first_index, second_index) if a != b})
    first_index, second_index = zip(*pairs)
    weights = pmi[first_index, second_index].reshape(-1)
    if name_map is not None:
        first_index = [name_map[i] for i in first_index]
        second_index = [name_map[i] for i in second_index]
    result = pd.DataFrame(dict(first=first_index, second=second_index, pmi=weights))
    result.sort_values('pmi', ascending=False, inplace=True)
    result.reset_index(inplace=True, drop=True)
    return result
