from wiki_dump_reader import Cleaner, iterate
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from collections import Counter
from scipy.sparse import csc_matrix, csr_matrix, save_npz, load_npz
from sklearn.preprocessing import normalize
import numpy as np
import pickle

class SearchNormalized:

    def __init__(self, files_number=50002, characters_per_file=-1):
        #self.bag_of_words = set()
        self.bag_of_words = None
        #self.file_dictionaries = []
        self.file_dictionaries = None
        self.texts = []
        self.files_number = files_number
        self.characters_per_file = characters_per_file
        self.term_by_document = None
        self.pattern = re.compile('[^a-z0-9]+')
        self.stop_words = set(stopwords.words('english'))

    def load_files(self):
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

    def load_preprocessed(self):
        with open("matrices/bag_of_words.txt", "rb") as fp:
            self.bag_of_words = pickle.load(fp)

        with open("matrices/file_dicts.txt", "rb") as fp:
            self.file_dictionaries = pickle.load(fp)

    def create_term_by_document_matrix(self):
        files = []
        words = []
        values = []

        #self.bag_of_words = list(self.bag_of_words)
        nw_vector = [0]*len(self.bag_of_words)
        for (f, file) in enumerate(self.file_dictionaries):
            for (w, word) in enumerate(self.bag_of_words):
                files.append(f)
                words.append(w)
                if word in file.keys():
                    values.append(file[word])
                    nw_vector[w]+=1
                else:
                    values.append(0)

        self.term_by_document = csc_matrix((values, (words, files))) #?
        nw_vector = np.array(nw_vector)
        idf = np.log(np.array([self.files_number]*len(self.bag_of_words))/nw_vector)
        idf_sparse_matrix = csc_matrix(idf.reshape(-1, 1))
        self.term_by_document = self.term_by_document.multiply(idf_sparse_matrix)
        self.term_by_document = normalize(self.term_by_document, norm='l2', axis=0, copy=False)

    def save(self):
        save_npz('matrices/normalized_term_by_document.npz', self.term_by_document)
        # with open("matrices/n_bag_of_words.txt", "wb") as fp:
        #     pickle.dump(self.bag_of_words, fp)
        # with open("matrices/n_texts.txt", "wb") as fp:
        #     pickle.dump(self.texts, fp)

    def load(self):
        self.term_by_document = load_npz('matrices/normalized_term_by_document.npz')
        with open("matrices/bag_of_words.txt", "rb") as fp:
            self.bag_of_words = pickle.load(fp)
        with open("matrices/texts.txt", "rb") as fp:
            self.texts = pickle.load(fp)

    def search(self, query, k):
        word_tokens = self.pattern.sub(' ', query.lower()).split(' ')
        cleaned_query = [PorterStemmer().stem(w) for w in word_tokens if w not in self.stop_words]
        query_dictionary = Counter(cleaned_query)

        query_indices = []
        for (w, word) in enumerate(self.bag_of_words):
            if word in query_dictionary.keys():
                query_indices.append(w)

        query_vector_transposed = csr_matrix((list(query_dictionary.values()), ([0]*len(query_indices), query_indices)), (1, len(self.bag_of_words)))
        query_vector_transposed = normalize(query_vector_transposed, norm='l2', axis=1, copy=False)

        similarity = query_vector_transposed * self.term_by_document

        #print(similarity)

        return np.argpartition(-1*similarity.toarray().squeeze(), k)[:k]

if __name__ == '__main__':
    s = SearchNormalized()
    s.load_preprocessed()
    s.create_term_by_document_matrix()
    s.save()
    #print(len(s.bag_of_words))

