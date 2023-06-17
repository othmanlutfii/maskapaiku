import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, send_file
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import base64
from PIL import Image
from io import BytesIO
import os
os.chdir(r'C:\Users\othma\PycharmProjects\UAS NLPp (2)\UAS NLPp')
import flask

app = Flask(__name__)


# model = pickle.load(open('./model/model.sav','rb'))

def cleaning(df):
    import re
    import string

    def clean(text):
        text = str(text)
        text = text.lower()
        text = re.sub("@[A-Za-z0-9_]+", "", text)  # Menghapus @<name> [mention twitter]
        text = re.sub("#\w+", "", text)
        text = re.sub("\[.*?\]", "", text)
        text = re.sub("https?://\S+|www\.\S+", "", text)
        text = re.sub("<.*?>+", "", text)
        text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
        # text = re.sub('\n', '', text)
        text = re.sub("\w*\d\w*", "", text)
        text = re.sub("\d+", "", text)
        text = re.sub("\s+", " ", text).strip()
        # text = re.sub('\n', '', text) jadi:
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text

    # Applying the cleaning function
    df['tweet2'] = df['Tweet'].apply(lambda x: clean(x))
    df.head()

    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    import swifter

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    df['cleaned'] = df['tweet2'].swifter.apply(lambda x: stemmer.stem(x))

    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

    factory = StopWordRemoverFactory()

    stopword = factory.get_stop_words()
    stopwords = factory.create_stop_word_remover()

    df['cleaned'] = df['cleaned'].apply(lambda x: stopwords.remove(x))

    from nltk import word_tokenize
    from nltk.corpus import stopwords

    new_stopwords = ['sahabat', 'nomor', 'yg', 'jd', 'klo', 'pa', 'bpj', 'nya', 'tp', 'ga', 'jg', 'https', 'co', 'aja',
                     'ya',
                     'gw', 'kalo', 'tuh', 'tau', 'gk', 'gak', 'kalo', 'amp', 'gitu', 'krn', 'dr', 'sih', 'gue', 'bgt',
                     'aja', 'ya', 'krn', 'pake', 'udah', 'sampe', 'udah', 'emang', 'nggak', 'gk', 'udh', 'kela',
                     'bpjskesehatanri', 'duanya', 'banget', 'tdk', 'bpjsmelayani', 'gotongroyongsemuatertolong',
                     'gotongroyongsemuatertolong_jknhadiruntukrakyat', 'bpjskesehatan_bpjsmelayani'
        , 'dgn', 'nih', 'loh', 'dpt', 'yaa', 'dah', 'kak', 'sm', 'ngga', 'dg', 'deh', 'lho', 'utk', 'kali', 'a', 'b',
                     'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                     'w', 'x', 'y', 'z', 'gua', 'sya', 'iya', 'ni']

    stop_words = stopwords.words('indonesian')
    stop_words.extend(new_stopwords)

    def removeStopword(str):
        stop_words = stopwords.words('indonesian')
        stop_words.extend(new_stopwords)
        # stop_words = set(stopwords.words('indonesian'))
        word_tokens = word_tokenize(str)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return ' '.join(filtered_sentence)

    df['cleaned'] = df['cleaned'].apply(lambda x: removeStopword(x))

    return df


def scraping(maskapai):
    from datetime import date
    from dateutil.relativedelta import relativedelta
    import snscrape.modules.twitter as sntwitter

    # Creating list to append tweet data to

    attributes_container = []

    # Using TwitterS earchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(
            f'{maskapai} since:{date.today() + relativedelta(months=-2)} until:{date.today()} lang:id').get_items()):  # big mouth:2022-08-05 until:2022-09-06
        if i > 100:
            break
        attributes_container.append(
            [tweet.user.username, tweet.id, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])

    # Creating a dataframe to load the list

    tweet = pd.DataFrame(attributes_container,
                         columns=["User", "Tweet ID", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"])

    tweet.to_csv(f'{maskapai}.csv', index=False)

    return tweet


def load_model_predict(df):
    sentences = df['cleaned'].tolist()

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 1000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 50
    # This is fixed.
    EMBEDDING_DIM = 16

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['Tweet'].values)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df['Tweet'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    from tensorflow.keras.models import model_from_json

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('model.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    y_pred = loaded_model.predict(X)

    df['Sentiment'] = np.NaN
    for i in range(len(y_pred)):
        df['Sentiment'][i] = np.argmax(y_pred[i])

    df = df.replace([0.0, 1.0, 2.0], ['negatif', 'netral', 'positif'])

    return df

def wordcloud(df, sentimen, maskapai, title):

    try:
        df = df[(df['Sentiment'] == sentimen)]


        # Import the wordcloud library
        from wordcloud import WordCloud
        # Join the different processed titles together.
        long_string = ','.join(map(str, list(df['cleaned'].values)))
        # Create a WordCloud object
        
        word_cloud = WordCloud(
                width=400,
                height=400,
                random_state=1,
                background_color="white",
                colormap="Pastel1",
                collocations=False,
                ).generate(long_string)
        
        import random

        def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
            h = 344
            s = int(100.0 * 255.0 / 255.0)
            l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
            return "hsl({}, {}%, {}%)".format(h, s, l)
        print(word_cloud)
        fig = plt.figure(1)
        plt.imshow(word_cloud.recolor(color_func= random_color_func, random_state=3),
                interpolation="bilinear")
        plt.title(title)
        plt.axis('off')
        plt.savefig(f'./static/output_files/{sentimen}{maskapai}.jpg')
    except:
        pass
    
def lda_predict(df, sentimen, maskapai):
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.models.ldamodel import LdaModel
    from gensim.corpora.dictionary import Dictionary
    from gensim import  models
    from wordcloud import WordCloud
       
        
    text = df.cleaned
    text_list =  [i.split() for i in text]

    
    from gensim import corpora

    dictionary = corpora.Dictionary(text_list)


    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
   
    tfidf = models.TfidfModel(doc_term_matrix) #build TF-IDF model
    corpus_tfidf = tfidf[doc_term_matrix]
    
    lda_model = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=4) #num topic menyesuaikan hasil dari coherence value paling tinggi
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic)) 
        
    import matplotlib.colors as mcolors    
    from collections import Counter
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in text_list for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(8,5), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)    
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
    plt.savefig(f'static/output_files/topic{maskapai}{sentimen}.jpg')




@app.route('/')
def index():
    return render_template("index.html", hasil="")

@app.route('/garuda')
def garuda():
    df = scraping("garuda indonesia")
    df = load_model_predict(cleaning(df))
    df['no'] = np.arange(len(df))
    x = df[df['Sentiment'] == 'positif'].value_counts()
    positif = sum(x)
    y = df[df['Sentiment'] == 'negatif'].value_counts()
    negatif = sum(y)
    z = df[df['Sentiment'] == 'netral'].value_counts()
    netral = sum(z)
    Items = [(a, b, c) for a, b, c in zip(df['no'], df['Tweet'], df['Sentiment'])]
    import os, shutil
    folder = './static/output_files'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
    
    
    wordcloud(df, 'positif', 'garuda', 'POSITIF')
    wordcloud(df, 'negatif', 'garuda', 'NEGATIF')
    wordcloud(df, 'netral', 'garuda', 'NETRAL')

    lda_predict(df,'positif', 'garuda')

    data = {'Sentiment': 'Count', 'Positive': positif, 'Negative': negatif, 'Netral': netral}
    return render_template('garuda.html',items=Items,dashboardPie=data)

@app.route('/sriwijaya')
def sriwijaya():
    df = scraping("Sriwijaya Air")
    df = load_model_predict(cleaning(df))
    df['no'] = np.arange(len(df))
    x = df[df['Sentiment'] == 'positif'].value_counts()
    positif = sum(x)
    y = df[df['Sentiment'] == 'negatif'].value_counts()
    negatif = sum(y)
    z = df[df['Sentiment'] == 'netral'].value_counts()
    netral = sum(z)
    Items = [(a, b, c) for a, b, c in zip(df['no'], df['Tweet'], df['Sentiment'])]
    import os, shutil
    folder = './static/output_files'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    wordcloud(df, 'positif', 'sriwijaya', 'POSITIF')
    wordcloud(df, 'negatif', 'sriwijaya', 'NEGATIF')
    wordcloud(df, 'netral', 'sriwijaya', 'NETRAL')
  
    lda_predict(df,'positif', 'sriwijaya')

    data = {'Sentiment': 'Count', 'Positive': positif, 'Negative': negatif, 'Netral': netral}
    return render_template('sriwijaya.html',items=Items,dashboardPie=data)

@app.route('/lion')
def lion():
    df = scraping("Lion Air")
    df = load_model_predict(cleaning(df))
    df['no'] = np.arange(len(df))
    x = df[df['Sentiment'] == 'positif'].value_counts()
    positif = sum(x)
    y = df[df['Sentiment'] == 'negatif'].value_counts()
    negatif = sum(y)
    z = df[df['Sentiment'] == 'netral'].value_counts()
    netral = sum(z)
    Items = [(a, b, c) for a, b, c in zip(df['no'], df['Tweet'], df['Sentiment'])]
    data = {'Sentiment': 'Count', 'Positive': positif, 'Negative': negatif, 'Netral': netral}
    import os, shutil
    folder = './static/output_files'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    wordcloud(df, 'positif', 'lion', 'POSITIF')
    wordcloud(df, 'negatif', 'lion', 'NEGATIF')
    wordcloud(df, 'netral', 'lion', 'NETRAL')
    
    lda_predict(df,'positif', 'lion')
    
    return render_template('lion.html',items=Items,dashboardPie=data)

@app.route('/citilink')
def citilink():
    df = scraping("Citilink")
    df = load_model_predict(cleaning(df))
    df['no'] = np.arange(len(df))
    x = df[df['Sentiment'] == 'positif'].value_counts()
    positif = sum(x)
    y = df[df['Sentiment'] == 'negatif'].value_counts()
    negatif = sum(y)
    z = df[df['Sentiment'] == 'netral'].value_counts()
    netral = sum(z)
    import os, shutil
    folder = './static/output_files'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    Items = [(a, b, c) for a, b, c in zip(df['no'], df['Tweet'], df['Sentiment'])]
    data = {'Sentiment': 'Count', 'Positive': positif, 'Negative': negatif, 'Netral': netral}

    lda_predict(df,'positif', 'citilink')

    wordcloud(df, 'positif', 'citilink', 'POSITIF')
    wordcloud(df, 'negatif', 'citilink', 'NEGATIF')
    wordcloud(df, 'netral', 'citilink', 'NETRAL')
    return render_template('citilink.html',items=Items,dashboardPie=data)

@app.route('/batik')
def batik():
    df = scraping("Batik Air")
    df = load_model_predict(cleaning(df))
    df['no'] = np.arange(len(df))
    x = df[df['Sentiment'] == 'positif'].value_counts()
    positif = sum(x)
    y = df[df['Sentiment'] == 'negatif'].value_counts()
    negatif = sum(y)
    z = df[df['Sentiment'] == 'netral'].value_counts()
    netral = sum(z)
    Items = [(a, b, c) for a, b, c in zip(df['no'], df['Tweet'], df['Sentiment'])]
    data = {'Sentiment': 'Count', 'Positive': positif, 'Negative': negatif, 'Netral': netral}
    import os, shutil
    folder = './static/output_files'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    lda_predict(df,'positif', 'batik')
    wordcloud(df, 'positif', 'batik', 'POSITIF')
    wordcloud(df, 'negatif', 'batik', 'NEGATIF')
    wordcloud(df, 'netral', 'batik', 'NETRAL')
  

    return render_template('batik.html',items=Items,dashboardPie=data)




if __name__ == "__main__":
    app.run(debug=True)
