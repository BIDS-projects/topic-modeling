class DocumentItem():
    def __init__(self, base_url):
        self.base_url = base_url
        self.document = list()

    def add_words(self, word_list):
        assert type(word_list) == list, "List needed"
        #self.document.extend(word_list.split())
        self.document.extend(word_list)

    #@property
    def get_document(self):
        return self.document
    #@property
    def get_base_url(self):
        return self.base_url
