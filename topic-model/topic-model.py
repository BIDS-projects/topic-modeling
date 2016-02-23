## What is dirichlet distribution?
# https://www.hakkalabs.co/articles/the-dirichlet-distribution

## Term weighting: tf-idf (term frequency-inverse document frequency)
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# TF-IDF gives important words
# run separately on each documents?
# feed the results into LDA?

## Introduction to Probabilistic Topic Model
# https://www.cs.princeton.edu/~blei/papers/Blei2012.pdf

"""LDA analysis of BIDS data.

Usage:
  lda.py [--tiers=<max_tier>] [--fn=<fn>] [--alpha=<param>] [--ratio=<ratio>]

Options:
  -h --help           Display usage.
  --alpha=<param>     Set the weighting function parameter. [default: 1.0]
  --fn=<fn>           Choose the weighting function. [default: poisson_law]
  --tiers=<max_tier>  Set the max tier. [default: 3]
  --ratio=<ratio>     Sets the summarizer ratio. [default: 0.5]

"""
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from time import time
from util import DocumentItem, MongoDB_loader
import models
import reduction
import numpy as np

# From http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#example-applications-topics-extraction-with-nmf-lda-py


def write_to_file(filename, uc):
    with open(filename, 'wb') as f:
        f.write(uc.encode('utf8'))


class TopicModel():
    n_topics = 6
    n_top_words = 10

    def __init__(self, max_degree=3, fn=models.poisson_law, alpha=1.0, ratio=0.5):
        self.MAX_DEGREE = max_degree
        self.fn = fn
        self.alpha = alpha
        self.ratio = ratio
        print("Importing data...")
        document_items = MongoDB_loader().get_corpus()
        reduction_obj = reduction.Reduction()
        self.inst_urls = []
        self.inst_names = []
        self.data_samples = []
        for item in document_items:
            # TODO: process to save into inst_names
            self.inst_urls.append(item.get_base_url())
            # removes numbers
            print("Filtering: {}".format(item.get_base_url()))
            document = self.filter_document(item.get_document())
            #document = item.get_document()

            #reduced_document = " ".join(reduction_obj.reduce(document, self.ratio))

            self.data_samples.append(document)
            #for _ in range(self.apply_weighting(item.get_tier())):
            #   #self.data_samples.append(reduced_document)
            #   self.data_samples.append(document)


    def filter_document(self, document):
        filtered = []
        custom_stopwords = ['div']
        for word in document.split():
            #if 'div' in word.lower():
                #return "1"
            if not (word.isdigit() or word[0] == '-' and word[1:].isdigit()) and (word.lower() not in custom_stopwords):
                filtered.append(word)
        return " ".join(filtered)

    def apply_weighting(self, tier):
        """
        Takes in the degree of separation of a document and applies a weighting function.
        """
        return int(self.fn(tier, self.alpha) / self.fn(self.MAX_DEGREE, self.alpha))

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print("\t" + "\n\t".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print

    def lda_analysis(self, ngram_low, ngram_high):
        print("Extracting tf features for LDA...")
        #tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf_vectorizer = CountVectorizer(ngram_range=(ngram_low, ngram_high), max_df=0.95, min_df=2, stop_words='english')
        t0 = time()
        tf = tf_vectorizer.fit_transform(self.data_samples)
        print("done in %0.3fs." % (time() - t0))
        n_samples, n_features = tf.get_shape()

        print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..." % (n_samples, n_features))
        lda = LatentDirichletAllocation(n_topics=self.n_topics, max_iter=5,
                                        learning_method='online', learning_offset=50.,
                                        random_state=0)
        t0 = time()
        lda.fit(tf)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in LDA model:")
        tf_feature_names = tf_vectorizer.get_feature_names()
        self.print_top_words(lda, tf_feature_names, self.n_top_words)

    def nmf_analysis(self, ngram_low, ngram_high):
        # TF-IDF + NMF
        # Use tf-idf features for NMF.
        print("Extracting tf-idf features for NMF...")
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(ngram_low,ngram_high),max_df=0.95, min_df=2, #max_features=n_features,
                                        stop_words='english')
        t0 = time()
        tfidf = tfidf_vectorizer.fit_transform(self.data_samples)
        tfidf_dtm = tfidf.toarray()
        print("done in %0.3fs." % (time() - t0))

        # AFTER FITTING VOCAB
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        tfidf_vocab = np.array(tfidf_feature_names)

        # Fit the NMF model
        print("Fitting the NMF model with tf-idf features,")
    #        "n_samples=%d and n_features=%d..."
    #        % (n_samples, n_features))
        t0 = time()
        nmf = NMF(n_components=self.n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
        nmf_clf = NMF(n_components=self.n_topics, random_state=1, alpha=.1, l1_ratio=.5)
        doctopic = nmf_clf.fit_transform(tfidf_dtm)

        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in NMF model:")
        topic_words = []
        for topic in nmf_clf.components_:
            word_idx = np.argsort(topic)[::-1][0:self.n_top_words]
            topic_words.append([tfidf_vocab[i] for i in word_idx])
        doctopic_prop = doctopic / np.sum(doctopic, axis=1, keepdims=True)
        self.print_top_words(nmf, tfidf_feature_names, self.n_top_words)

        inst_urls = np.asarray(self.inst_urls)
        doctopic_orig = doctopic_prop.copy()

        n_groups = len(set(inst_urls))
        doctopic_grouped = np.zeros((n_groups, self.n_topics))
        for i, name in enumerate(sorted(set(inst_urls))):
            doctopic_grouped[i, :] = np.mean(doctopic_prop[inst_urls == name, :], axis=0)

        insts = sorted(set(inst_urls))
        for i in range(len(doctopic_grouped)):
            top_topics = np.argsort(doctopic_grouped[i,:])[::-1][0:3]
            top_topics_str = ' '.join(str(t) for t in top_topics)
            print("{}: {}".format(insts[i], top_topics_str))


if __name__ == "__main__":
    arguments = docopt(__doc__)
    max_degree = int(arguments['--tiers'])
    fn = getattr(models, arguments['--fn'])
    alpha = float(arguments['--alpha'])
    ratio = float(arguments['--ratio'])
    tm = TopicModel(max_degree, fn, alpha, ratio)
    #tm.lda_analysis(2,3)
    tm.nmf_analysis(2,3)

    ##### PLAYING WITH TF-IDF #####
    #tf_vectorizer = CountVectorizer(ngram_range=(1,3), max_df=0.95, min_df=2, stop_words='english')
    #tf = tf_vectorizer.fit_transform(tm.data_samples)
    #write_to_file("tf.log", "\n".join(tf_vectorizer.get_feature_names()))

    #tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.95, min_df=2, #max_features=n_features,
    #                                   stop_words='english')
    #tfidf = tfidf_vectorizer.fit_transform(tm.data_samples)
    #write_to_file("tfidf.log", "\n".join(tfidf_vectorizer.get_feature_names()))
