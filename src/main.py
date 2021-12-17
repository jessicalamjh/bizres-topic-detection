import gensim

import json
import os
from pprint import pprint
from tqdm import tqdm

def read_jsonl(filepath):
    # read multi-line json file
    assert os.path.exists(filepath)
    
    data = []
    with open(filepath, 'r') as f:
        for line in tqdm(f):
            x = json.loads(line)
            data.append(x)
    return data

###############################################################################
# load data and paragraphs
data = read_jsonl(
    "../data/selfparsed/preprocessed-english-other-noaliases.jsonl")
data += read_jsonl(
    "../data/selfparsed/preprocessed-english-sustainability-noaliases.jsonl")
print(f"Total number of texts: {len(data)}")

paragraphs = [para for x in data for para in x['preprocessed']]
print(f"Total number of paragraphs: {len(paragraphs)}\n")

# build corpus of bigrams and dictionary
bigram_model = gensim.models.phrases.Phrases(paragraphs, min_count=10)
corpus = [bigram_model[para] for para in paragraphs]

id2word = gensim.corpora.Dictionary(corpus)
print(f"Original size of dictionary: {len(id2word)}")
id2word.filter_extremes(no_below=50, no_above=0.1, keep_n=50000)
print(f"Size of filtered dictionary: {len(id2word)}")

# get BoW embeddings
word_ids = [id2word.doc2bow(para) for para in corpus]

# validate for num_topics
scores = []
for num_topics in [10]:
	lda_model = gensim.models.ldamodel.LdaModel(
        corpus=word_ids,
        id2word=id2word,
        num_topics=num_topics, 
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True,
    )

	coherence_model_lda = gensim.models.CoherenceModel(
        model=lda_model, 
        texts=corpus, 
        dictionary=id2word, 
        coherence='c_v',
    )

	pprint(lda_model.print_topics())

    # outpath = f"../output/topics_{num_topics}.txt"
	with open(outpath, "w") as outfile:
		for line in lda_model.print_topics():
			outfile.write(str(line[0]) + "\t" + line[1] + "\n")
            scores.append(coherence_model.get_coherence())