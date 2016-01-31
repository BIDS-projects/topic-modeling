import numpy as np
import textmining
import lda
from pymongo import MongoClient
from items import DocumentItem


def createDocTermMat(dataset):
    """
    Transform the dataset from mockobjects as specified in test_lda.py in tests folder to a document-term matrix.
    
    A document-term matrix is a matrix of documents vs terms present in the collection of documents. 
    The entries denote the frequency of a term in a given document.
    
    Returns a triple of the data matrix, list of terms present in the corpus, and document titles.
    
    Need to find a way to cleanly distinguish between documents, i.e. need labels. Needs to be unique. 
    Note that the identifiers depend on what we choose as our definition for documents. For now this is all the text in a web directory. Future studies: research paper, papers of researchers, summary of institution, institutional research papers.
    """
    docTermMatrix = textmining.TermDocumentMatrix()
    titles = []
    # Adds each document to the docTermMatrix, assuming document is a list of words
    for document in dataset:
        titles.append(document.base_url)
        docTermMatrix.add_doc(document.text)
    temp = list(docTermMatrix.rows(cutoff=1))
    terms = tuple(temp[0])
    matrix = np.array(temp[1:])
    return [matrix, terms, titles]

def saveTo(name, dtm):
    """
        Save the document-term matrix to name. Cutoff represents the minimum frequency a word must have before it is considered
    """
    dtm.write_csv(name, cutoff=1)

class LDAM:
    def __init__(self, number_topics):
        self.model = lda.LDA(n_topics=number_topics, n_iter=1000, alpha = .05, eta = .005)
    
    def fit(self, dataset):
        [dataMatrix, terms, documents] = createDocTermMat(dataset)
        self.terms = terms
        self.documents = documents
        self.model.fit(dataMatrix)

    def printTopics(self, n_words):
        """
        Prints the top n_words for each topic.

        Things to think about: A way to visualize the distribution of these words in each topic.
        """
        for line in self.topics(n_words):
            print(line)

    def topics(self, n_words):
        """Gives a generator of values to print"""
        yield self.terms
        for i, top_dist in enumerate(self.model.topic_word_):
            topic_words = np.array(self.terms)[np.argsort(top_dist)][:-(n_words+1)]
            yield 'Topic {}: {}'.format(i, ' '.join(topic_words))

    def printDocTopic(self):
        """
        Prints the document in order indicates which topic it is most likely under.

        Possible additions: Add the top three most likely topics. Indicate the likelihood.
        Things to think about: Look for documents that may be allocated to different topics. i.e. look at the likelihood and if it surpasses a certain threshold for more than one topic indicate it. This could lead to further study.
        """
        doc_topic = self.model.doc_topic_
        for i in range(len(self.documents)):
            print("{} (top topic: {})".format(self.documents[i], doc_topic[i].argmax()))



class MongoDB_loader():
    def __init__():
        settings = {'MONGODB_SERVER':"localhost",
                    'MONGODB_PORT': 27017,
                    'MONGODB_DB': "ecosystem_mapping",
                    'MONGODB_LINK_COLLECTION': "link_collection",
                    'MONGODB_TEXT_COLLECTION': "text_collection"}

        connection = MongoClient(
            settings['MONGODB_SERVER'],
            settings['MONGODB_PORT']
        )
        db = connection[settings['MONGODB_DB']]
        selftext_collection = db[settings['MONGODB_TEXT_COLLECTION']]


    def filter_words(word_string):
        return word_string

    def get_corpus():
        uniq_base_urls = text_collection.distinct("base_url")
        corpus = []
        for base_url in uniq_base_urls:
            item = DocumentItem(base_url)
            for data in text_collection.find({"base_url": base_url}):
                item.add_words(filter_words(data['text']))
            corpus.append(item)
        return corpus
