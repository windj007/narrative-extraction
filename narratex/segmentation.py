import gensim
import numpy as np


def calc_topic_entropy(topic_proba_pairs):
    return -sum(p * np.log2(p) for _, p in topic_proba_pairs)


GOOD_POS = {'VERB', 'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN'}


def calc_topics_sim(a, b):
    anorm = sum([p ** 2 for _, p in a]) ** 0.5
    bnorm = sum([p ** 2 for _, p in b]) ** 0.5
    adict = dict(a)
    bdict = dict(b)
    both_keys = set(adict) & set(bdict)
    return sum(adict[k] * bdict[k] for k in both_keys) / (anorm * bnorm)


def infer_segmentation(all_texts, model_window_size=2, num_topics=40, passes=10, segment_window_size=5):
    all_texts_with_tokens = [[[tok.lemma.lower() for tok in sent.joint
                               if tok.upos in GOOD_POS]
                              for sent in doc]
                             for doc in all_texts]
    model_chunks = []
    segment_chunks = []
    for doc in all_texts_with_tokens:
        for start_i in range(0, len(doc), model_window_size):
            cur_chunk = [tok
                         for sent in doc[start_i:start_i + model_window_size]
                         for tok in sent]
            model_chunks.append(cur_chunk)
        for start_i in range(0, len(doc), max(1, segment_window_size // 2)):
            cur_chunk = [tok
                         for sent in doc[start_i:start_i + segment_window_size]
                         for tok in sent]
            segment_chunks.append(cur_chunk)

    vocab = gensim.corpora.dictionary.Dictionary(model_chunks)
    vocab[0]

    model_chunk_bow = [vocab.doc2bow(ch) for ch in model_chunks]
    segment_chunk_bow = [vocab.doc2bow(ch) for ch in segment_chunks]

    topic_model = gensim.models.LdaMulticore(model_chunk_bow,
                                             num_topics=num_topics, passes=passes, id2word=vocab.id2token)

    segment_chunk_topics = [topic_model[ch] for ch in segment_chunk_bow]
    adj_sim = [calc_topics_sim(segment_chunk_topics[i], segment_chunk_topics[i + 1])
               for i in range(len(segment_chunk_topics) - 1)]

    # segment_chunk_entropy = [calc_topic_entropy(tops) for tops in segment_chunk_topics]
    # print('min', min(segment_chunk_entropy), 'max', max(segment_chunk_entropy), np.mean(segment_chunk_entropy))
    print('min', min(adj_sim), 'max', max(adj_sim), 'mean', np.mean(adj_sim))
    for i in range(100):
        print(segment_chunks[i])
        print(adj_sim[i])
        print()
