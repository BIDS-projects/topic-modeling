class DocumentItem():
    def __init__(self, base_url):
        self.base_url = base_url
        self.document = list()
    def add_words(self, word_list):
        self.document.extend(word_list)

    @property
    def document(self):
        return self.document
    @property
    def base_url(sefl):
        return self.base_url
