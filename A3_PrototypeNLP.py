# MA5851: DATA SCIENCE MASTERCLASS 1
# A3: WebCrawler and NLP System
# Task 3: Prototype NLP

# Code blocks:
# 1. Import the Required Packages
# 2. Natural Language Pre-Processing
# 3. NLP Task: Keyword Extraction with Yake
# 4. NLP Task: Sentiment Analysis using Vader
# 5. Post NLP Analysis


##### 1. Import the Required Packages
import pandas as pd
from bs4 import BeautifulSoup
import unidecode
from word2number import w2n
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import yake
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


##### 2. Natural Language Pre-Processing
# Sources:
# - https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79
# - https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html
# - https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html

# import the harvested munged data
df = pd.read_csv('munged_df.csv')
# copy the description field as a new variable
df['desc_clean'] = df['description']

# A) remove HTML tags
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text
df['desc_clean'] = df['desc_clean'].apply(strip_html_tags)

# B) remove accented characters
def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text
df['desc_clean'] = df['desc_clean'].apply(remove_accented_chars)

# C) lowercase all texts
def lower_case(text):
    return str(text).lower()
df['desc_clean'] = df['desc_clean'].apply(lower_case)

# D) expand contractions
def expand_contractions(text):
    text = contractions.fix(text)
    return text
df['desc_clean'] = df['desc_clean'].apply(expand_contractions)

# E) remove special characters (i.e., non-alphanumeric)
def irrelevant_characters(text):
    return re.sub('[^A-Za-z0-9 ]+', ' ', text)
df['desc_clean'] = df['desc_clean'].apply(irrelevant_characters)

# F) convert number words to numeric form
def text_to_numeric(text):
    s = text.split()
    o = []
    for word in s:
        try:
            o += [str(w2n.word_to_num(word))]
        except ValueError:
            o += [word]
    text_numeric = (' '.join(o))
    return text_numeric
df['desc_clean'] = df['desc_clean'].apply(text_to_numeric)

# G) remove numbers
def remove_numbers(text):
    new_text = "".join(filter(lambda x: not x.isdigit(), text))
    return new_text
df['desc_clean'] = df['desc_clean'].apply(remove_numbers)

# H) remove extra whitespaces
def remove_whitespace(text):
    text = text.strip()
    return " ".join(text.split())
df['desc_clean'] = df['desc_clean'].apply(remove_whitespace)

# I) tokenization
def make_tokens(text):
    return word_tokenize(text)
df['desc_clean'] = df['desc_clean'].apply(make_tokens)

# J) remove Stopwords
stop_words = set(stopwords.words('english'))
def clean_stopwords(token):
    return [item for item in token if item not in stop_words]
df['desc_clean'] = df['desc_clean'].apply(clean_stopwords)

# K) lemmatization
lemma = WordNetLemmatizer()
def clean_lemmatization(token):
    return [lemma.lemmatize(word=w, pos='v') for w in token]
df['desc_clean'] = df['desc_clean'].apply(clean_lemmatization)

# L) remove the words having length <= 3
def clean_length(token):
    return [i for i in token if len(i) > 3]
df['desc_clean'] = df['desc_clean'].apply(clean_length)

# M) convert the list of tokens into back to the string
def convert_to_string(listReview):
    return " ".join(listReview)
df['desc_clean'] = df['desc_clean'].apply(convert_to_string)


##### 3. NLP Task: Keyword Extraction with Yake
# specify the parameters
language = "en"                     # English
max_ngram_size = 1                  # Single word max (instead of 2=bigram or 3=trigram)
deduplication_thresold = 0.1        # 0.1 (avoid repetition) to 0.9 (allow repetition)
deduplication_algo = 'seqm'         # Structural Edge Quality Metric (over leve or jaro)
windowSize = 1                      # Computing words left to right
numOfKeywords = 50                  # Number of words to be extracted
custom_kw_extractor = yake.KeywordExtractor(lan=language,
                                            n=max_ngram_size,
                                            dedupLim=deduplication_thresold,
                                            dedupFunc=deduplication_algo,
                                            windowsSize=windowSize,
                                            top=numOfKeywords,
                                            features=None)
# extract the keywords
df['desc_yake'] = df['desc_clean'].apply(custom_kw_extractor.extract_keywords)
# convert the list of tuples returned back to a string
df['desc_yake'] = df['desc_yake'].astype(str)
# replace non-alphabet characters with a blank space
df['desc_yake'] = df['desc_yake'].str.replace('[^a-zA-Z]', ' ')
# remove excessive white spaces
df['desc_yake'] = df['desc_yake'].apply(remove_whitespace)


##### 4. NLP Task: Sentiment Analysis using Vader
analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(x):
    score = analyser.polarity_scores(x)
    return score
df['sentiment'] = df['desc_yake'].apply(lambda x: sentiment_analyzer_scores(x))
df['negative'] = df['sentiment'].apply(lambda score_dict: score_dict['neg'])
df['neutral'] = df['sentiment'].apply(lambda score_dict: score_dict['neu'])
df['positive'] = df['sentiment'].apply(lambda score_dict: score_dict['pos'])
df['compound'] = df['sentiment'].apply(lambda score_dict: score_dict['compound'])
df['comp_score'] = df['compound'].apply(lambda c: 'Positive' if c >=0 else 'Negative')

# convert and store as csv
df.to_csv('final_df.csv', index=False)


##### 5. Post NLP Analysis

# first split the df down the middle
Hi_df = df[df.rating_class == 'High']
Lo_df = df[df.rating_class == 'Low']

# second create a univatiate summary of the numeric fields
Hi_df[["desc_length", "avg_rating", "rating_count", "wavg_rating", "api_page_count",
       "negative", "neutral", "positive", "compound"]].describe().to_csv('Hi_uni.csv', index=True)
Lo_df[["desc_length", "avg_rating", "rating_count", "wavg_rating", "api_page_count",
       "negative", "neutral", "positive", "compound"]].describe().to_csv('Lo_uni.csv', index=True)

# third boxplots the vader sentiment fields by rating_class
# negative
print(df.groupby('rating_class')['negative'].median())
plt.figure(figsize=(6, 10))
sns.boxplot(df.rating_class, df.negative).set(ylim=(0, 1))
plt.savefig('class_by_neg.png', bbox_inches='tight')
plt.show()
# neutral
print(df.groupby('rating_class')['neutral'].median())
plt.figure(figsize=(6, 10))
sns.boxplot(df.rating_class, df.neutral).set(ylim=(0, 1))
plt.savefig('class_by_neu.png', bbox_inches='tight')
plt.show()
# positive
print(df.groupby('rating_class')['positive'].median())
plt.figure(figsize=(6, 10))
sns.boxplot(df.rating_class, df.positive).set(ylim=(0, 1))
plt.savefig('class_by_pos.png', bbox_inches='tight')
plt.show()
# compound
print(df.groupby('rating_class')['compound'].median())
plt.figure(figsize=(6, 10))
sns.boxplot(df.rating_class, df.compound).set(ylim=(-1, 1))
plt.savefig('class_by_comp.png', bbox_inches='tight')
plt.show()

# fourth create word frequency tables
def word_frequency(sentence):
    # joins all the sentences
    sentence = " ".join(sentence)
    # creates tokens
    new_tokens = word_tokenize(sentence)
    # counts the words
    counted = Counter(new_tokens)
    # creates and returns a data frame
    word_freq = pd.DataFrame(counted.items(),
                             columns=['word',
                                      'frequency']).sort_values(by='frequency',
                                                                       ascending=False)
    return word_freq

# plot the frequencies
# high performing
wf_hi = word_frequency(df.desc_yake[df.rating_class == 'High'])
wf_hi.to_csv('wf_hi.csv', index=False)
plt.figure(figsize=(10, 10))
sns.barplot(x='frequency', y='word', data=wf_hi.head(30))
plt.savefig('wf_hi.png', bbox_inches='tight')
plt.show()
# low performing
wf_lo = word_frequency(df.desc_yake[df.rating_class == 'Low'])
wf_lo.to_csv('wf_lo.csv', index=False)
plt.figure(figsize=(10, 10))
sns.barplot(x='frequency', y='word', data=wf_lo.head(30))
plt.savefig('wf_lo.png', bbox_inches='tight')
plt.show()
# isolate words only in high performing
common = wf_hi.merge(wf_lo, on=["word"])
common['frequency'] = (common['frequency_x'] + common['frequency_y'])
wf_unique = wf_hi[~wf_hi.word.isin(common.word)]
# drop those listed <3 times
wf_unique = wf_unique[wf_unique.frequency > 2]
# drop the obvious names
wf_unique = wf_unique[wf_unique.word != "green"]
wf_unique = wf_unique[wf_unique.word != "marcus"]
wf_unique = wf_unique[wf_unique.word != "geralt"]
wf_unique = wf_unique[wf_unique.word != "glokta"]
wf_unique = wf_unique[wf_unique.word != "jordan"]
wf_unique = wf_unique[wf_unique.word != "darrow"]
wf_unique = wf_unique[wf_unique.word != "stormbringer"]
wf_unique = wf_unique[wf_unique.word != "mistborn"]
wf_unique = wf_unique[wf_unique.word != "winterfell"]
wf_unique = wf_unique[wf_unique.word != "malaz"]
wf_unique = wf_unique[wf_unique.word != "roland"]
wf_unique = wf_unique[wf_unique.word != "angeles"]
wf_unique = wf_unique[wf_unique.word != "severian"]
wf_unique = wf_unique[wf_unique.word != "ibram"]
wf_unique = wf_unique[wf_unique.word != "vaelin"]
wf_unique = wf_unique[wf_unique.word != "sorna"]
wf_unique = wf_unique[wf_unique.word != "daenerys"]
wf_unique = wf_unique[wf_unique.word != "ophelia"]
wf_unique = wf_unique[wf_unique.word != "jardir"]
wf_unique = wf_unique[wf_unique.word != "raesinia"]
wf_unique = wf_unique[wf_unique.word != "vordan"]
wf_unique = wf_unique[wf_unique.word != "raif"]
wf_unique = wf_unique[wf_unique.word != "gwynne"]
wf_unique = wf_unique[wf_unique.word != "vivacia"]
wf_unique = wf_unique[wf_unique.word != "althea"]
wf_unique = wf_unique[wf_unique.word != "vestrit"]
wf_unique = wf_unique[wf_unique.word != "lannister"]
wf_unique = wf_unique[wf_unique.word != "america"]
wf_unique = wf_unique[wf_unique.word != "lamora"]
wf_unique = wf_unique[wf_unique.word != "alba"]
wf_unique = wf_unique[wf_unique.word != "jake"]
wf_unique = wf_unique[wf_unique.word != "eddie"]
wf_unique = wf_unique[wf_unique.word != "anomander"]
wf_unique = wf_unique[wf_unique.word != "tiste"]
wf_unique = wf_unique[wf_unique.word != "geder"]
wf_unique = wf_unique[wf_unique.word != "cithrin"]
wf_unique = wf_unique[wf_unique.word != "falcio"]
wf_unique = wf_unique[wf_unique.word != "gavin"]
wf_unique = wf_unique[wf_unique.word != "murgen"]
wf_unique = wf_unique[wf_unique.word != "taglios"]
wf_unique = wf_unique[wf_unique.word != "cale"]
wf_unique = wf_unique[wf_unique.word != "fitz"]
wf_unique = wf_unique[wf_unique.word != "cantor"]
wf_unique = wf_unique[wf_unique.word != "richard"]
wf_unique = wf_unique[wf_unique.word != "jezal"]
wf_unique = wf_unique[wf_unique.word != "kovacs"]
wf_unique = wf_unique[wf_unique.word != "takeshi"]
wf_unique = wf_unique[wf_unique.word != "sull"]
wf_unique = wf_unique[wf_unique.word != "susannah"]
wf_unique = wf_unique[wf_unique.word != "cardan"]
wf_unique = wf_unique[wf_unique.word != "jude"]
wf_unique = wf_unique[wf_unique.word != "orboan"]
wf_unique = wf_unique[wf_unique.word != "palliako"]
wf_unique = wf_unique[wf_unique.word != "wester"]
wf_unique = wf_unique[wf_unique.word != "marko"]
wf_unique = wf_unique[wf_unique.word != "kennit"]
wf_unique = wf_unique[wf_unique.word != "bingtown"]
wf_unique = wf_unique[wf_unique.word != "teresa"]
wf_unique = wf_unique[wf_unique.word != "clara"]
wf_unique = wf_unique[wf_unique.word != "kalliam"]
wf_unique = wf_unique[wf_unique.word != "dalinar"]
wf_unique = wf_unique[wf_unique.word != "stormlight"]
wf_unique = wf_unique[wf_unique.word != "mexico"]
wf_unique = wf_unique[wf_unique.word != "trollslayer"]
wf_unique = wf_unique[wf_unique.word != "thrax"]
wf_unique = wf_unique[wf_unique.word != "terminus"]
wf_unique = wf_unique[wf_unique.word != "casca"]
wf_unique = wf_unique[wf_unique.word != "corban"]
wf_unique = wf_unique[wf_unique.word != "ciri"]
wf_unique = wf_unique[wf_unique.word != "arlen"]
wf_unique = wf_unique[wf_unique.word != "baru"]
wf_unique = wf_unique[wf_unique.word != "kellanved"]
wf_unique = wf_unique[wf_unique.word != "dean"]
wf_unique = wf_unique[wf_unique.word != "needle"]
wf_unique = wf_unique[wf_unique.word != "fillory"]
wf_unique = wf_unique[wf_unique.word != "quentin"]
wf_unique = wf_unique[wf_unique.word != "elim"]
wf_unique = wf_unique[wf_unique.word != "esslemont"]
wf_unique = wf_unique[wf_unique.word != "girton"]
wf_unique = wf_unique[wf_unique.word != "satrapies"]
wf_unique = wf_unique[wf_unique.word != "kyle"]
wf_unique = wf_unique[wf_unique.word != "chromeria"]
wf_unique = wf_unique[wf_unique.word != "edur"]
wf_unique = wf_unique[wf_unique.word != "lannisters"]
wf_unique = wf_unique[wf_unique.word != "china"]
wf_unique = wf_unique[wf_unique.word != "joffrey"]
wf_unique = wf_unique[wf_unique.word != "davarus"]
wf_unique = wf_unique[wf_unique.word != "corvere"]
wf_unique = wf_unique[wf_unique.word != "tamas"]
wf_unique = wf_unique[wf_unique.word != "kylar"]
wf_unique = wf_unique[wf_unique.word != "essun"]
wf_unique = wf_unique[wf_unique.word != "esmenet"]
wf_unique = wf_unique[wf_unique.word != "darujhistan"]
wf_unique = wf_unique[wf_unique.word != "ringil"]
wf_unique = wf_unique[wf_unique.word != "tyrion"]
wf_unique = wf_unique[wf_unique.word != "asin"]
wf_unique = wf_unique[wf_unique.word != "tavore"]
wf_unique = wf_unique[wf_unique.word != "savine"]
wf_unique = wf_unique[wf_unique.word != "corelings"]
wf_unique = wf_unique[wf_unique.word != "brock"]
wf_unique = wf_unique[wf_unique.word != "poppy"]
wf_unique = wf_unique[wf_unique.word != "nikan"]

# fifth create wordclouds from the high, low and unique word frequencies
# https://stackoverflow.com/questions/57826063/how-to-create-a-wordcloud-according-to-frequencies-in-a-pandas-dataframe
# convert the frequency tables to dictionaries
wf_hi_dic = wf_hi.set_index('word').to_dict()['frequency']
wf_lo_dic = wf_lo.set_index('word').to_dict()['frequency']
wf_uni_dic = wf_unique.set_index('word').to_dict()['frequency']
wf_com_dic = common.set_index('word').to_dict()['frequency']
# generate the wordclouds
# High performing
wc_hi = WordCloud(width=800,
                  height=400,
                  max_words=100,
                  background_color="white").generate_from_frequencies(wf_hi_dic)
plt.figure()
plt.imshow(wc_hi, interpolation="bilinear")
plt.axis("off")
plt.savefig('wc_hi.png', bbox_inches='tight')
plt.show()
# Low performing
wc_lo = WordCloud(width=800,
                  height=400,
                  max_words=100,
                  background_color="white").generate_from_frequencies(wf_lo_dic)
plt.figure()
plt.imshow(wc_lo, interpolation="bilinear")
plt.axis("off")
plt.savefig('wc_lo.png', bbox_inches='tight')
plt.show()
# Unique to High
wc_uni = WordCloud(width=800,
                   height=400,
                   max_words=100,
                   background_color="white").generate_from_frequencies(wf_uni_dic)
plt.figure()
plt.imshow(wc_uni, interpolation="bilinear")
plt.axis("off")
plt.savefig('wc_uni.png', bbox_inches='tight')
plt.show()
# Common to both
wc_com = WordCloud(width=800,
                   height=400,
                   max_words=100,
                   background_color="white").generate_from_frequencies(wf_com_dic)
plt.figure()
plt.imshow(wc_com, interpolation="bilinear")
plt.axis("off")
plt.savefig('wc_com.png', bbox_inches='tight')
plt.show()








