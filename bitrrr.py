import pattern
from pattern.web import URL, DOM, plaintext

from os import listdir
from os.path import isfile, join
import os

import pandas as pd
import numpy as np
import nltk
from collections import Counter, defaultdict
from nltk.corpus import stopwords
import string
import random

filedic = {}
for each in os.listdir('lyrics'):
    if each != '.DS_Store':
        filedic[each] = listdir('lyrics/'+each)[1:]

lyricsdic = defaultdict(list)
exclude = set(string.punctuation)
lyricslist = []

remove_list = ['verse','hook','chorus', '','choruslloyd banks', '50 cent', 'outro','bridge', 'chorus', 'lyrics', 'lloyd banks', 'rick ross','lil wayne', 'da', 'dem', 'im', 'ya', 'dat', 'tony yayo', 'banks yay 50',
              'verse 1 tony yayo', 'verse 4 lloyd banks', 'verse 5 50 cent', 'verse 2 tony yayo', 'verse 1 50 cent', 'chorus 50 cent', 'intro 50 cent', 'biggie sample  in the background throughout the song', 'verse 3  tony yayo', 'verse 1  50 cent','chorus  50 cent','verse 2  lloyd banks','verse 2 lloyd banks', 'verse 6 tony yayo', 'verse 4 yony yayo']

for k,v in filedic.iteritems():
    for link in v:
        if link != '.ipynb_checkpoints':
            page = open('lyrics/'+k+'/'+link).read()
            dom = DOM(page)
            lyrics = dom('.lyrics')[0]
            p = plaintext(lyrics.content)
            p = ''.join(ch for ch in p if ch not in exclude)
            lyricslist.append(p)
            p = p.splitlines()
            lyricsdic[k].append(filter(lambda x: x.lower() not in remove_list, p))


    lineslist = []
remove_list = ['verse','hook','chorus', '','choruslloyd banks', '50 cent', 'outro','bridge', 'chorus', 'lyrics', 'lloyd banks', 'rick ross','lil wayne', 'da', 'dem', 'im', 'ya', 'dat', 'tony yayo', 'banks yay 50',
              'verse 1 tony yayo', 'verse 4 lloyd banks', 'verse 5 50 cent', 'verse 2 tony yayo', 'verse 1 50 cent', 'chorus 50 cent', 'intro 50 cent', 'biggie sample  in the background throughout the song', 'verse 3  tony yayo', 'verse 1  50 cent','chorus  50 cent','verse 2  lloyd banks','verse 2 lloyd banks', 'verse 6 tony yayo', 'verse 4 yony yayo']
exportlist = []
for each in lyricslist:
    lineslist.append(each.splitlines())
lineslist = np.array(lineslist)
for each in lineslist:
    exportlist.append(filter(lambda x: x.lower() not in remove_list, each))

tokenlist = []

for lyrics in lyricslist[:100]:
    tokens = nltk.word_tokenize(lyrics)
    tokens = [w for w in tokens if w.lower() not in ['50','cent','chief','keef', 'lloyd', 'banks', 'busta','rhymes']]
    tokenlist.extend([w for w in tokens if w.lower() not in stopwords.words('english')])

import random

class Markov(object):

    def __init__(self, words):
        self.cache = {}
        #self.open_file = open_file
        self.words = words
        self.word_size = len(self.words)
        self.database()

    def file_to_words(self):
        self.open_file.seek(0)
        data = self.open_file.read()
        words = data.split()
        return words

    def triples(self):
        if len(self.words) < 3:
            return
        for i in range(len(self.words) - 2):
            yield (self.words[i], self.words[i+1], self.words[i+2])

    def database(self):
        for w1, w2, w3 in self.triples():
            key = (w1, w2)
            if key in self.cache:
                self.cache[key].append(w3)
            else:
                self.cache[key] = [w3]

    def generate_markov_text(self, size=25):
        seed = random.randint(0, self.word_size-3)
        seed_word, next_word = self.words[seed], self.words[seed+1]
        w1, w2 = seed_word, next_word
        gen_words = []
        for i in xrange(size):
            gen_words.append(w1)
            w1, w2 = w2, random.choice(self.cache[(w1, w2)])
        gen_words.append(w2)
        return ' '.join(gen_words)

mark = Markov(tokenlist)
mark.generate_markov_text(size = 50)


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

vectorizer = TfidfVectorizer(stop_words='english', max_df = .6)
X = vectorizer.fit_transform(lyricslist)
features = vectorizer.get_feature_names()
kmeans = KMeans()
kmeans.fit(X)

top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-6:-1]
print "top features for each cluster:"
for num, centroid in enumerate(top_centroids):
    print "%d: %s" % (num, ", ".join(features[i] for i in centroid))

def rhyme(inp, level):
    entries = nltk.corpus.cmudict.entries()
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return set(rhymes)

tokenlines = []
wordtypes = defaultdict(list)

for each in lyricsdic['mfdoom']:
    for line in each:
        tokens = nltk.word_tokenize(line)
        struct = nltk.pos_tag(tokens)
        for each in struct:
            wordtypes[each[1]].append(each[0])
        tokenlines.append(','.join([x[1] for x in struct]))

sf = []
for each in lyricsdic['mfdoom'][0]:
    l = nltk.word_tokenize(each)
    t = nltk.pos_tag(l)
    sf.extend([x[1] for x in t])
    sf.extend('\n ')
sf = ' '.join(sf)
for each in sf.splitlines():
    sentence = ""
    for a in each.split():
        sentence +=  random.choice(wordtypes[a]).lower() + " "
    print sentence

Counter(tokenlines).most_common(25)
sentence = ""

for each in ['PRP','VBD','DT','NN','PRP','VBD','DT','NN']:
    sentence +=  random.choice(wordtypes[each]).lower() + " "

print sentence
# last_word = sentence.rsplit(None, 1)[-1]
print ""

for each in ['PRP$','NNS','CC','PRP$','NN','NNP']:
     print random.choice(wordtypes[each]).lower(),

postag = []
for line in tokenlines[0:10]:
    tags = nltk.pos_tag(line)
    #print tags
    #postag.append([[x[1] for x in tags]])

Counter(tokenlines).most_common(10)
sentence = ""

for each in ['IN','DT','NN','PRP','VBP','PRP','VBP','JJ','NNS']:
    sentence +=  random.choice(wordtypes[each]).lower() + " "

print sentence
last_word = sentence.rsplit(None, 1)[-1]
print ""

for each in ['PRP','VBP','WRB','TO','VB','NN']:
    print random.choice(wordtypes[each]).lower(),