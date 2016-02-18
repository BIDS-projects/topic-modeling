class DocumentItem():
    def __init__(self, base_url):
        self.base_url = base_url
        self.document = list()
        self.deg_sep = float('inf') # Should be updated to the correct degree.

    def add_words(self, word_list):
        assert type(word_list) == list, "List needed"
        #self.document.extend(word_list.split())
        self.document.extend(word_list)

    def update_degree(self, degree):
        self.deg_sep = min(degree, self.deg_sep)

    # @property
    def get_document(self):
        return self.document

    # @property
    def get_base_url(self):
        return self.base_url

    # @property
    def get_deg_sep(self):
        return self.deg_sep
