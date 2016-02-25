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
                text = data['text']
                tier = data['tier']
                # TODO: Exclude .XML pages
                corpus.append(DocumentItem(base_url, text, tier))
        return corpus
