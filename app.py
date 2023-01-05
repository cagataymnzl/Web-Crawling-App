from flask import Flask, render_template, url_for, request, send_file, Response
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
import nltk
import textblob as tb
from subprocess import check_output
from wordcloud import WordCloud,STOPWORDS
from tqdm import tqdm
from bokeh.plotting import figure, show
from bokeh.embed import components
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.autolayout"] = True
news_feed = pd.read_csv('input/news-week-18aug24-mini.csv', dtype={'publish_time': object})
news_feed['publish_hour'] = news_feed.publish_time.str[:10]
news_feed['publish_date'] = news_feed.publish_time.str[:8]
news_feed['publish_hour_only'] = news_feed.publish_time.str[8:10]
news_feed['publish_time_only'] = news_feed.publish_time.str[8:12]
days = news_feed['publish_date'].unique().tolist()

news_feed['dt_time'] = pd.to_datetime(news_feed['publish_time'], format='%Y%m%d%H%M')
news_feed['dt_hour'] = pd.to_datetime(news_feed['publish_hour'], format='%Y%m%d%H')
news_feed['dt_date'] = pd.to_datetime(news_feed['publish_date'], format='%Y%m%d')
feed_count = news_feed['feed_code'].value_counts()
feed_count = feed_count[:10, ]
news_feed = news_feed.dropna()
englishStopWords = set(nltk.corpus.stopwords.words('english'))
nonEnglishStopWords = set(nltk.corpus.stopwords.words()) - englishStopWords
stopWordsDictionary = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template("index.html")

@app.route('/statistics', methods=['GET'])
def statistics():
    return render_template('statistics.html')

@app.route('/show-top-10-feeds')
def plot_top_10_feeds():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    xs = feed_count.index
    ys = feed_count.values

    axis.bar(xs, ys)
    axis.set_title("TOP 10 FEEDS")
    axis.set_xlabel("Feed Code")
    axis.set_ylabel("No of Occurances")

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

news_feed.headline_text.dropna()

def get_language(text):
    if type(text) is str:
        text = text.lower()
    words = set(nltk.wordpunct_tokenize(text))
    return max(((lang, len(words & stopwords)) for lang, stopwords in stopWordsDictionary.items()), key = lambda x: x[1])[0]

news_feed['language'] = news_feed['headline_text'].apply(get_language)
language_count = news_feed['language'].value_counts()
language_count = language_count[:10]

@app.route('/show-top-10-languages')
def plot_top_10_languages():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    xs = language_count.index
    ys = language_count.values

    axis.bar(xs, ys)
    axis.set_title("TOP 10 LANGUAGES")
    axis.set_xlabel("Language")
    axis.set_ylabel("No of Occurances")

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    
@app.route('/cloud', methods=['GET'])
def cloud():
    news_feed_english_df = news_feed[news_feed['language'] == 'english']
    news_feed_english = news_feed_english_df['headline_text']
    words = ' '.join(news_feed_english)
    cleaned_word = " ".join([word for word in words.split()])
    wordcloud = WordCloud(stopwords = STOPWORDS,
                         background_color = 'black',
                         width = 2500,
                         height = 2500
                         ).generate(cleaned_word)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.imshow(wordcloud)
    axis.axis('off')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

news_feed.headline_text.dropna()
news_feed_english_df = news_feed[news_feed['language'] == 'english']
news_feed_english = news_feed_english_df['headline_text']
    
def display_topics(model, feature_names, no_top_words):
    output = " "
    for topic_idx , topic in enumerate(model.components_):
        output = output + "\nTopic %d:" % (topic_idx) + "\n"
        output = output +  " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words -1:-1]])        
    return output

@app.route('/nmf-results', methods=['GET', 'POST'])
def nmf():
    #non-negative matrix factorization NMF, 
    #büyük miktardaki verileri, metin verilerini örneğin, 
    #verilerin boyutsallığını azaltan daha küçük, daha seyrek gösterimleri azaltmak için kullanılabilir 
    #(aynı bilgiler çok daha az değişken kullanılarak korunabilmektedir).
    
    no_features = (int) (request.form['no_of_features'])

    tfidf_vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 2, max_features=no_features, stop_words = 'english')
    tfidf = tfidf_vectorizer.fit_transform(news_feed_english)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    no_topic = (int) (request.form['no_of_topics'])
    nmf = NMF(n_components=no_topic, random_state = 1, l1_ratio=.5, init = 'nndsvd').fit(tfidf)
    no_top_words = (int) (request.form['no_of_top_words'])

    #display_topics(nmf ,tfidf_feature_names, no_top_words)
    print("********************************************************")
    #display_topics(lda , tf_feature_names , no_top_words)
    return render_template('nmf.html',output1=display_topics(nmf ,tfidf_feature_names, no_top_words))

@app.route('/lda-results', methods=['GET', 'POST'])
def lda():
    #Latent Dirichlet allocation (LDA), doğal dil işlemede kullanılan her belgenin 
    #bir konu koleksiyonu kabul edildiği ve belgedeki her kelimenin konulardan birine karşılık geldiği 
    #en basit kabul edilen bir konu modelleme örneğidir.

    no_features = (int) (request.form['no_of_features'])

    tf_vectorizer = CountVectorizer(max_df = 0.95, min_df = 2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(news_feed_english)
    tf_feature_names = tf_vectorizer.get_feature_names_out()

    no_topic = (int) (request.form['no_of_topics'])
    lda = LatentDirichletAllocation(n_components=no_topic, max_iter = 5, learning_method = 'online', learning_offset=50., random_state=0).fit(tf)
    no_top_words = (int) (request.form['no_of_top_words'])

    #display_topics(nmf ,tfidf_feature_names, no_top_words)
    print("********************************************************")
    #display_topics(lda , tf_feature_names , no_top_words)
    return render_template('lda.html',output2=display_topics(lda , tf_feature_names , no_top_words)
)

@app.route('/statistics', methods=['GET'])
def show_statistics():
    return render_template('statistics.html')  

@app.route('/time-polarities', methods=['GET'])
def show_time_polarities():
    news_feed.headline_text.dropna()
    news_feed_english_df = news_feed[news_feed['language'] == 'english']
    def sent(x):
        t = tb.TextBlob(x)
        return t.sentiment.polarity, t.sentiment.subjectivity
    tqdm.pandas(leave = False, mininterval = 25)
    vals = news_feed_english_df.headline_text.progress_apply(sent)

    news_feed_english_df['polarity'] = vals.str[0]
    news_feed_english_df['sub'] = vals.str[1]
    
    fig = Figure()
    mean_pol = list(dict(news_feed_english_df.groupby('dt_time')['polarity'].mean()).items())
    mean_pol.sort(key=lambda x: x[0])
    axis = fig.add_subplot(2, 2, 1)
    axis.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])
    axis.set_title("Mean polarity over time")
    #----------------------------------------------------------------------------------------------------
    
    mean_pol = list(dict(news_feed_english_df.groupby('dt_time')['sub'].mean()).items())
    mean_pol.sort(key=lambda x: x[0])
    axis = fig.add_subplot(2, 2, 2)
    axis.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])
    axis.set_title("Mean subjectivity over time")
    #----------------------------------------------------------------------------------------------------
    
    mean_pol = list(dict(news_feed_english_df.groupby('dt_time')['polarity'].std()).items())
    mean_pol.sort(key=lambda x: x[0])
    axis = fig.add_subplot(2, 2, 3)
    axis.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])
    axis.set_title("Std Dev of polarity over time")
    #----------------------------------------------------------------------------------------------------
    
    mean_pol = list(dict(news_feed_english_df.groupby('dt_time')['sub'].std()).items())
    mean_pol.sort(key=lambda x: x[0])
    axis = fig.add_subplot(2, 2, 4)
    axis.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])
    axis.set_title("Std dev of subjectivity over time")
    #----------------------------------------------------------------------------------------------------
    
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    
@app.route('/hour-polarities', methods=['GET'])
def show_hour_polarities():
    news_feed.headline_text.dropna()
    news_feed_english_df = news_feed[news_feed['language'] == 'english']
    def sent(x):
        t = tb.TextBlob(x)
        return t.sentiment.polarity, t.sentiment.subjectivity
    tqdm.pandas(leave = False, mininterval = 25)
    vals = news_feed_english_df.headline_text.progress_apply(sent)

    news_feed_english_df['polarity'] = vals.str[0]
    news_feed_english_df['sub'] = vals.str[1]
    
    fig = Figure()
    mean_pol = list(dict(news_feed_english_df.groupby('dt_hour')['polarity'].mean()).items())
    mean_pol.sort(key=lambda x: x[0])
    axis = fig.add_subplot(2, 2, 1)
    axis.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])
    axis.set_title("Mean polarity over time")
    #----------------------------------------------------------------------------------------------------
    
    mean_pol = list(dict(news_feed_english_df.groupby('dt_hour')['sub'].mean()).items())
    mean_pol.sort(key=lambda x: x[0])
    axis = fig.add_subplot(2, 2, 2)
    axis.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])
    axis.set_title("Mean subjectivity over time")
    #----------------------------------------------------------------------------------------------------
    
    mean_pol = list(dict(news_feed_english_df.groupby('dt_hour')['polarity'].std()).items())
    mean_pol.sort(key=lambda x: x[0])
    axis = fig.add_subplot(2, 2, 3)
    axis.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])
    axis.set_title("Std Dev of polarity over time")
    #----------------------------------------------------------------------------------------------------
    
    mean_pol = list(dict(news_feed_english_df.groupby('dt_hour')['sub'].std()).items())
    mean_pol.sort(key=lambda x: x[0])
    axis = fig.add_subplot(2, 2, 4)
    axis.plot([i[0] for i in mean_pol], [i[1] for i in mean_pol])
    axis.set_title("Std dev of subjectivity over time")
    #----------------------------------------------------------------------------------------------------
    
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
      
if __name__ == "__main__":
    app.run(debug=True)
