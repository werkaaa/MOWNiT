from wiki_dump_reader import Cleaner, iterate
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from collections import Counter
from scipy.sparse import csc_matrix, csr_matrix, save_npz, load_npz
from sklearn.preprocessing import normalize
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
import operator


class SearchNoIDF:

    def __init__(self, files_number=50002, characters_per_file=-1):
        self.bag_of_words = set()
        self.file_dictionaries = []
        self.texts = []
        self.files_number = files_number
        self.characters_per_file = characters_per_file
        self.term_by_document = None
        self.pattern = re.compile('[^a-z0-9]+')
        self.stop_words = set(stopwords.words('english'))
        self.svd = None
        self.svd_term_space = None
        self.query_vector_transposed = None
        self.dictionary = {}
        self.nw_vector = []
        self.svd_components = None

    def load_files(self, dictionary_size=20000):
        cleaner = Cleaner()
        i = 0
        for title, text in iterate('wiki/simplewiki-20191120-pages-articles.xml'):
            if i >= self.files_number:
                break
            cleaned_text = cleaner.clean_text(text)[:self.characters_per_file]
            cleaned_fragment, _ = cleaner.build_links(text)
            self.texts.append(title)

            word_tokens = self.pattern.sub(' ', cleaned_text.lower()).split(' ')
            cleaned_text = [PorterStemmer().stem(w) for w in word_tokens if w not in self.stop_words]
            self.file_dictionaries.append(Counter(cleaned_text))
            self.bag_of_words = self.bag_of_words.union(set(cleaned_text))
            i += 1

        self.dictionary = {w: 0 for w in self.bag_of_words}
        for file in self.file_dictionaries:
            for word in self.bag_of_words:
                if word in file.keys():
                    self.dictionary[word] += 1

        if len(self.dictionary) > dictionary_size:
            self.dictionary = Counter(self.dictionary).most_common(dictionary_size)
            self.bag_of_words = []
            for (word, num) in self.dictionary:
                self.bag_of_words.append(word)
                self.nw_vector.append(num)
        else:
            self.bag_of_words = list(self.dictionary.keys())
            self.nw_vector = list(self.dictionary.values())

    def create_term_by_document_matrix(self):
        files = []
        words = []
        values = []

        self.bag_of_words = list(self.bag_of_words)

        for (f, file) in enumerate(self.file_dictionaries):
            for (w, word) in enumerate(self.bag_of_words):
                if word in file.keys():
                    files.append(f)
                    words.append(w)
                    values.append(file[word])

        self.term_by_document = csc_matrix((values, (words, files)))
        self.term_by_document = normalize(self.term_by_document, norm='l2', axis=0, copy=False)

    def save(self):
        save_npz(f'matrices{self.files_number}_noidf/normalized_term_by_document.npz', self.term_by_document)
        with open(f'matrices{self.files_number}_noidf/svd_bag_of_words.txt', "wb") as fp:
            pickle.dump(self.bag_of_words, fp)
        with open(f'matrices{self.files_number}_noidf/svd_texts.txt', "wb") as fp:
            pickle.dump(self.texts, fp)

    def save_svd(self):
        with open(f'matrices{self.files_number}_noidf/svd_components.txt', "wb") as fp:
            pickle.dump(self.svd_components, fp)
        with open(f'matrices{self.files_number}_noidf/svd_term_space.txt', "wb") as fp:
            pickle.dump(self.svd_term_space, fp)

    def load(self):
        self.term_by_document = load_npz(f'matrices{self.files_number}_noidf/normalized_term_by_document.npz')
        with open(f'matrices{self.files_number}_noidf/svd_bag_of_words.txt', "rb") as fp:
            self.bag_of_words = pickle.load(fp)
        with open(f'matrices{self.files_number}_noidf/svd_texts.txt', "rb") as fp:
            self.texts = pickle.load(fp)

    def load_svd(self):
        with open(f'matrices{self.files_number}_noidf/svd_term_space.txt', "rb") as fp:
            self.svd_term_space = pickle.load(fp)
        with open(f'matrices{self.files_number}_noidf/svd_components.txt', "rb") as fp:
            self.svd_components = pickle.load(fp)

    def create_query_vector_transposed(self, query):
        word_tokens = self.pattern.sub(' ', query.lower()).split(' ')
        cleaned_query = [PorterStemmer().stem(w) for w in word_tokens if w not in self.stop_words]
        query_dictionary = Counter(cleaned_query)

        query_words = []
        query_indices = []
        for (w, word) in enumerate(self.bag_of_words):
            if word in query_dictionary.keys():
                query_indices.append(w)
                query_words.append(query_dictionary[word])

        if len(query_indices) == 0:
            self.query_vector_transposed = None
            return

        self.query_vector_transposed = csr_matrix(
            (query_words, ([0] * len(query_indices), query_indices)), (1, len(self.bag_of_words)))
        self.query_vector_transposed = normalize(self.query_vector_transposed, norm='l2', axis=1, copy=False)

    def search(self, query, k):
        self.create_query_vector_transposed(query)
        if self.query_vector_transposed is None:
            return 'No similarity found'

        similarity = self.query_vector_transposed * self.term_by_document
        similarity = similarity.toarray().squeeze()

        indices = np.argpartition(-1 * similarity, k)[:k]
        indices_corr = [(i, similarity[i]) for i in indices]
        indices_corr.sort(key=operator.itemgetter(1), reverse=True)
        return indices_corr

    def get_matrix_svd(self, n_components=100):
        self.svd = TruncatedSVD(n_components=n_components)
        self.svd.fit(self.term_by_document)
        self.svd_term_space = self.svd.transform(self.term_by_document)
        self.svd_components = self.svd.components_

    def get_correlation_of_query_svd(self, query):
        self.create_query_vector_transposed(query)
        if self.query_vector_transposed is None:
            return None
        q_svd = self.query_vector_transposed.dot(self.svd_term_space)
        similarity = q_svd.dot(self.svd_components)
        return similarity

    def search_svd(self, query, k):
        similarity = self.get_correlation_of_query_svd(query)
        if similarity is None:
            return 'No similarity found'
        similarity = similarity.squeeze()
        indices = np.argpartition(-1 * similarity, k)[:k]
        indices_corr = [(i, similarity[i]) for i in indices]
        indices_corr.sort(key=operator.itemgetter(1), reverse=True)
        return indices_corr


# if __name__ == '__main__':
#     s = SearchNoIDF()
#     s.load_files()
#     s.create_term_by_document_matrix()
#     s.save()
#     s.load()
#     print(s.search("4th month april", 3))
