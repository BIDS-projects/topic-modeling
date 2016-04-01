from pymongo import MongoClient


class DocumentItem():

    def __init__(self, base_url, text, tier):
        """Initialize item."""
        self.base_url = base_url
        self.text = text
        self.tier = tier


class MongoDBLoader():

    def __init__(self):
        """Initializes the connection."""
        settings = {'MONGODB_SERVER':"localhost",
                    'MONGODB_PORT': 27017,
                    'MONGODB_DB': "ecosystem_mapping",
                    'MONGODB_COLLECTION': "filtered_collection"}

        connection = MongoClient(
            settings['MONGODB_SERVER'],
            settings['MONGODB_PORT']
        )
        db = connection[settings['MONGODB_DB']]
        self.collection = db[settings['MONGODB_COLLECTION']]

    def get_corpus(self):
        """Reads the objects from the database."""
        unique_base_urls = self.collection.distinct("base_url")
        corpus = []
        for base_url in unique_base_urls:
            for data in self.collection.find({"base_url": base_url}):
                if ".xml" not in data['src_url']:
                    text = data['text']
                    tier = data['tier']
                    corpus.append(DocumentItem(base_url, text, tier))
        return corpus

    def locate_keywords(self, keywords, head=10):
        """Locates the URLs in which the keyword phrase is found."""
        unique_src_urls = self.collection.distinct("src_url")
        matches = []
        for src_url in unique_src_urls:
            for data in self.collection.find({"src_url": src_url}):
                hits = data['text'].count(keywords)
                if hits > 0:
                    matches.append((src_url, hits))
        matches.sort(key=lambda (src_url, hits): -hits)
        for src_url, hits in matches:
            print str(hits).rjust(5), src_url
            head -= 1
            if head <= 0:
                break
