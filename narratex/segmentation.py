import gensim


def infer_segmentation(all_texts, window_size=5, num_topics=20, passes=5):
    all_texts_with_tokens = [[[tok.lemma.lower() for tok in sent.joint] for sent in doc]
                             for doc in all_texts]
    chunks = []
    for doc in all_texts_with_tokens:
        for start_i in range(0, len(doc), window_size // 2):
            cur_chunk = [tok
                         for sent in doc[start_i:start_i + window_size]
                         for tok in sent]
            chunks.append(cur_chunk)
    print('chunks', chunks[:1])

    vocab = gensim.corpora.dictionary.Dictionary(chunks)
    chunk_bow = [vocab.doc2bow(ch) for ch in chunks]
    topic_model = gensim.models.LdaModel(chunk_bow, num_topics=num_topics, id2word=vocab.id2token, passes=passes)
    vectors = [topic_model[ch] for ch in chunk_bow]
    print(vectors[:1])



