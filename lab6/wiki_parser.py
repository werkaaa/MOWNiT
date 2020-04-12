from wiki_dump_reader import Cleaner, iterate
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import re
cleaner = Cleaner()
files_number = 50002
i = 0
titles = []
bag_of_words = set()
file_dictionaries = []
pattern = re.compile('[^a-z0-9]+')
stop_words = set(stopwords.words('english'))
for title, text in iterate('wiki/simplewiki-20191120-pages-articles.xml'):
    if i >= files_number:
        break
    titles.append(title)
    cleaned_text = cleaner.clean_text(text)
    #cleaned_fragment, _ = cleaner.build_links(cleaned_text)
    # f = open(f'wiki/files/{i}.txt', "w")
    # f.write(cleaned_fragment)
    # f.close()
    word_tokens = pattern.sub(' ', cleaned_text.lower()).split(' ')
    cleaned_text = [PorterStemmer().stem(w) for w in word_tokens if w not in stop_words]
    file_dictionaries.append(Counter(cleaned_text))
    bag_of_words = bag_of_words.union(set(cleaned_text))
    i += 1

with open("matrices/texts.txt", "wb") as fp:
    pickle.dump(titles, fp)

with open("matrices/bag_of_words.txt", "wb") as fp:
    pickle.dump(list(bag_of_words), fp)

with open("matrices/file_dicts.txt", "wb") as fp:
    pickle.dump(file_dictionaries, fp)

