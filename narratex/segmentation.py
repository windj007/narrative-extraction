import gensim
import numpy as np


def calc_topic_entropy(topic_proba_pairs):
    return -sum(p * np.log2(p) for _, p in topic_proba_pairs)


GOOD_POS = {'VERB', 'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN'}


def infer_segmentation(all_texts, window_size=5, num_topics=40, passes=10):
    all_texts_with_tokens = [[[tok.lemma.lower() for tok in sent.joint
                               if tok.upos in GOOD_POS]
                              for sent in doc]
                             for doc in all_texts]
    chunks = []
    for doc in all_texts_with_tokens:
        for start_i in range(0, len(doc), window_size // 2):
            cur_chunk = [tok
                         for sent in doc[start_i:start_i + window_size]
                         for tok in sent]
            chunks.append(cur_chunk)

    vocab = gensim.corpora.dictionary.Dictionary(chunks)
    vocab[0]
    chunk_bow = [vocab.doc2bow(ch) for ch in chunks]
    topic_model = gensim.models.LdaMulticore(chunk_bow, num_topics=num_topics, passes=passes, id2word=vocab.id2token)
    chunk_topics = [topic_model[ch] for ch in chunk_bow]
    chunk_entropy = [calc_topic_entropy(tops) for tops in chunk_topics]
    print('min', min(chunk_entropy), 'max', max(chunk_entropy), np.mean(chunk_entropy))
    print(chunk_entropy)




