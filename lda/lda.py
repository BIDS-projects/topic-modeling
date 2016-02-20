## What is dirichlet distribution?
# https://www.hakkalabs.co/articles/the-dirichlet-distribution

## Term weighting: tf-idf (term frequency-inverse document frequency)
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# TF-IDF gives important words
# run separately on each documents?
# feed the results into LDA?

## Introduction to Probabilistic Topic Model
# https://www.cs.princeton.edu/~blei/papers/Blei2012.pdf

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from time import time
from util import DocumentItem, MongoDB_loader

# From http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#example-applications-topics-extraction-with-nmf-lda-py

def write_to_file(filename, uc):
    f = open(filename, 'w')
    f.write(uc.encode('utf8'))
    f.close()

class TopicModel():
    n_topics = 3
    n_top_words = 10

    def __init__(self):
        print("Importing data...")
        document_items = MongoDB_loader().get_corpus()
        self.data_samples = []
        for item in document_items:
            # removes numbers
            #document = [word for word in item.get_document().split() if not (word.isdigit() or word[0] == '-' and word[1:].isdigit())]
            document = item.get_document()

            self.data_samples.append(document)

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print("\t" + "\n\t".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print

    def lda_analysis(self, ngram_low, ngram_high):
        assert ngram_low <= ngram_high, "ngram_low {} <= ngram_high {}".format(ngram_low, ngram_high)
        # ngram_range_low, ngram_range_high
        print("Extracting tf features for LDA...")
        #tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf_vectorizer = CountVectorizer(ngram_range=(ngram_low,ngram_high), max_df=0.95, min_df=2, stop_words='english')
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
        assert ngram_low <= ngram_high, "ngram_low {} <= ngram_high {}".format(ngram_low, ngram_high)
        # TF-IDF + NMF
        # Use tf-idf features for NMF.
        print("Extracting tf-idf features for NMF...")
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(ngram_low,ngram_high),max_df=0.95, min_df=2, #max_features=n_features,
                                        stop_words='english')
        t0 = time()
        tfidf = tfidf_vectorizer.fit_transform(self.data_samples)
        print("done in %0.3fs." % (time() - t0))


        # Fit the NMF model
        print("Fitting the NMF model with tf-idf features,")
    #        "n_samples=%d and n_features=%d..."
    #        % (n_samples, n_features))
        t0 = time()
        nmf = NMF(n_components=self.n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in NMF model:")
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        self.print_top_words(nmf, tfidf_feature_names, self.n_top_words)

if __name__ == "__main__":
    tm = TopicModel()
    tm.lda_analysis(2, 3)
    tm.nmf_analysis(2, 3)

    ##### PLAYING WITH TF-IDF #####
    #tf_vectorizer = CountVectorizer(ngram_range=(1,3), max_df=0.95, min_df=2, stop_words='english')
    #tf = tf_vectorizer.fit_transform(tm.data_samples)

    #tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.95, min_df=2, #max_features=n_features,
    #                                   stop_words='english')
    #tfidf = tfidf_vectorizer.fit_transform(tm.data_samples)
    #write_to_file("tf.log", "\n".join(tf_vectorizer.get_feature_names()))
    #write_to_file("tfidf.log", "\n".join(tfidf_vectorizer.get_feature_names()))
