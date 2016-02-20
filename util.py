from pymongo import MongoClient

class DocumentItem():
    def __init__(self, base_url):
        self.base_url = base_url
        self.document = []
        self.tier = float('inf')

    def add_words(self, text):
        # text could be unicode or string
        #assert type(word_list) == list, "List needed. Got {} instead".format(type(word_list))
        
        self.document.append(text)

    def update_tier(self, tier):
        self.tier = min(self.tier, tier)

    #@property
    def get_document(self):
        return " ".join(self.document)

    def get_list_of_words(self):
        return self.document

    #@property
    def get_base_url(self):
        return self.base_url

    # @property
    def get_tier(self):
        return self.tier


class MongoDB_loader():
    def __init__(self):
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
        self.text_collection = db[settings['MONGODB_TEXT_COLLECTION']]

    def get_corpus(self):
        uniq_base_urls = self.text_collection.distinct("base_url")
        corpus = []
        for base_url in uniq_base_urls:
            item = DocumentItem(base_url)
            for data in self.text_collection.find({"base_url": base_url}):
                item.add_words(data['text'])
                item.update_tier(data['tier'])
            corpus.append(item)
        return corpus

