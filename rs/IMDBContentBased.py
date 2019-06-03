import pandas as pd
# Libraries for text preprocessing
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

class IMDBContentBased:

    def __init__(self):
        self.df_imdb = pd.read_csv('./datasets/IMDB-Movie-Data.csv')
        self.X = pd.DataFrame()
    # Most frequently occuring words
    def get_top_n_words(corpus, n=None):
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in
                      vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1],
                            reverse=True)
        return words_freq[:n]

    def  tf_id(self, column, max_features):
        df_imdb = self.df_imdb.copy()
        df_imdb["Word_count"] = df_imdb[column].apply(lambda x: len(str(x).split(" ")))
        stop_words = set(stopwords.words("english"))
        corpus = []
        for i in range(0, df_imdb.shape[0]):
            # Remove punctuations
            text = re.sub('[^a-zA-Z]', ' ', df_imdb[column][i])

            # Convert to lowercase
            text = text.lower()

            # remove tags
            text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

            # remove special characters and digits
            text = re.sub("(\\d|\\W)+", " ", text)

            ##Convert to list from string
            text = text.split()

            ##Stemming
            ps = PorterStemmer()
            # Lemmatisation
            lem = WordNetLemmatizer()
            text = [lem.lemmatize(word) for word in text if not word in
                                                                stop_words]
            text = " ".join(text)
            corpus.append(text)
        cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=max_features, ngram_range=(1, 4))
        X = cv.fit_transform(corpus)
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(X)
        tf_idf_X = tfidf_transformer.transform(X)
        return pd.DataFrame(tf_idf_X.toarray(), index=df_imdb.Title, columns=cv.get_feature_names())

    def item_description(self):

        desc_feats = self.tf_id("Description", 1000)
        self.X[desc_feats.columns] = desc_feats
        directors_feats = pd.get_dummies(self.df_imdb['Director'])*0.4
        directors_feats.index = self.df_imdb.Title
        self.X[directors_feats.columns] = directors_feats
        #title_feats = self.tf_id("Title", 1000)
        #directors_feats = self.tf_id("Director", 1000)
        #actors_feats = self.tf_id("Actors", 1000)
        #self.X[title_feats.columns] = title_feats
        #self.X[directors_feats.columns] = directors_feats
        #self.X[actors_feats.columns] = actors_feats
        print(self.X.isnull().values.any())

    def profile_learning(self):
        return None

    def filtering(self, filme, n=10):
        sim = cosine_similarity(self.X)
        sim = pd.DataFrame(sim, index=self.X.index, columns=self.X.index)
        return sim[filme].sort_values(ascending=False)[1:n]


cb = IMDBContentBased()
cb.item_description()
