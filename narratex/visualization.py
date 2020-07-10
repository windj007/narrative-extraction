import re

import brave
import networkx as nx

from IPython.display import display as jupyter_display
import matplotlib.pyplot as plt

import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch


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


def get_group_span(token_ids, token_spans):
    return (token_spans[min(token_ids) - 1][0],
            token_spans[max(token_ids) - 1][1])


def mark_subtree(token_spans, token_ids, entity_id_prefix, entity_type, relation_title):
    token_ids = sorted(token_ids)
    groups = []
    cur_group = [token_ids[0]]
    for t in token_ids[1:]:
        if t - cur_group[-1] == 1:
            cur_group.append(t)
        else:
            groups.append(cur_group)
            cur_group = [t]
    if cur_group:
        groups.append(cur_group)

    entities = [(f'{entity_id_prefix}_{i}',
                 entity_type,
                 [get_group_span(group, token_spans)])
                for i, group in enumerate(groups)]
    relations = []
    return entities, relations


def joint_markup_to_brat(sent, syntax=True, events=True):
    text, token_spans = detokenize_sentence([t.form for t in sent.joint])
    entities = []
    attributes = []
    for tok in sent.joint:
        entities.append((f'token_{tok.id}',
                         tok.upostag,
                         [token_spans[tok.id - 1]]))
    relations = []
    if syntax:
        for tok in sent.joint:
            if tok.head is not None and tok.head > 0:
                relations.append((f'dep_{len(relations)}',
                                  f'synt-{tok.deprel}',
                                  (('child', f'token_{tok.id}'),
                                   ('head', f'token_{tok.head}'))))
    if events:
        event_list = sent.get('events', [])
        for i, ev in enumerate(event_list):
            ev_ent, ev_rel = mark_subtree(token_spans, ev.action,
                                          f'act_{i}',
                                          'act',
                                          'event-action')
            entities.extend(ev_ent)
            relations.extend(ev_rel)

    return dict(text=text,
                entities=entities,
                relations=relations,
                attributes=attributes)


DEFAULT_BRAVE_MARKUP_DEFINITION = dict(
    entity_types=[
        dict(type='act',
             bgColor='#df5d07',
             borderColor='darken')
    ]
)


def brave_visualize_sent(sent, **kwargs):
    return brave.brave(joint_markup_to_brat(sent, **kwargs),
                       DEFAULT_BRAVE_MARKUP_DEFINITION)


class SentenceVis:
    def __init__(self, name2group, group2event, event2ds, docs, display=True):
        self.name2group = name2group
        self.group2event = group2event
        self.event2ds = event2ds
        self.docs = docs
        self.display = display

    def __call__(self, group_name, show_n=5, **kwargs):
        events = self.group2event[self.name2group[group_name]]
        result = []
        for ev in events[:show_n]:
            doc_i, sent_i = self.event2ds[ev.id]
            widget = brave_visualize_sent(self.docs[doc_i][sent_i], **kwargs)
            result.append(widget)
            if self.display:
                jupyter_display(widget)
        return result


def plot_event_graph(group_similarity, group2name, min_sim=0, figsize=(20, 20),
                     node_size=70, font_size=10, weight_pow=1, weigth_factor=1, layout_kwargs=dict(),
                     fname=None):
    graph = nx.Graph()

    for g1 in range(group_similarity.shape[0]):
        name1 = group2name[g1]
        for g2 in range(g1 + 1, group_similarity.shape[0]):
            sim = group_similarity[g1, g2]
            if sim < min_sim:
                continue
            name2 = group2name[g2]
            graph.add_edge(name1, name2, weight=(sim ** weight_pow) * weigth_factor)

    pos = nx.spring_layout(graph, **layout_kwargs)

    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)

    nx.draw_networkx_nodes(graph, pos, node_size=node_size, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=font_size, font_family='sans-serif', ax=ax)

    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname)

    return fig


def make_dendrogram_dict(pairwise_weights, group2name, method='single'):
    pdist = ssd.squareform(pairwise_weights)
    z = sch.linkage(pdist, method=method, optimal_ordering=True)
    cluster_id2dict = [dict(name=group2name[i]) for i in range(len(group2name))]
    for merge_i, (clust1, clust2, dist, size) in enumerate(z):
        child1, child2 = cluster_id2dict[clust1], cluster_id2dict[clust2]
        name1, name2 = child1['name'], child2['name']
        subname1 = name1.split(', ')[0] if clust1 < len(pairwise_weights) else name1
        subname2 = name2.split(', ')[0] if clust2 < len(pairwise_weights) else name2

        name = f'[{subname1}] [{subname2}]'

        cluster_id2dict.append(dict(name=name, children=[child1, child2]))

    return cluster_id2dict[-1]
