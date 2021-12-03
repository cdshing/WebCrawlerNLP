## A3: WebCrawler and NLP System 
### D1: Overview

In the field of novel marketing, McGinn (2020) describes the back cover blurb or elevator pitch as a combination of taglines and descriptions of characters and central themes that is a genuine deal closer for hard copy and electronic book sales. The purpose of this NLP prototype is to demonstrate a practical solution to sourcing publicly available metadata on novels by sub-genre and identify the building blocks of descriptions used to promote successful (highly rated) products among competitors. The market chosen to prototype is “grimdark”, which is a subgenre of fantasy.

The primary source of public novel data in this process is the Goodreads social cataloguing website, which provides a platform for individuals to track, rate and review books they read. With an emphasis on social, the site represents a community of consumers providing independent feedback just as TripAdvisor does for travel and accommodation.

The secondary data source, for those novels identified by web scraping the Goodreads “grimdark” shelf, is the Google Books API. This represents another publicly available metadata delivery asset that can be queried and extracted to supplement missing data and add additional metadata fields to enrich the NLP analysis planned.

The Python script used to harvest these data sources is outlined in [A3_WebScraper](https://github.com/cdshing/WebCrawlerNLP/blob/main/A3_WebScraper.py).

Once the chosen Goodreads fields have been scraped and supplemented with query driven Google Books API fields, the data frames will be merged before undergoing pre-processing data cleansing and finally stored as CVS.

Once the raw data is prepared, the derived “description” field will undergo natural language pre-processing to prepare it for analysis. With the end goal being to assist publishers in creating more successful back page blurbs, key words will then be extracted (using Yake) from the marketing heavy bag of words sourced on each identified title. As shown in the figure below, these identified key words will then be subjected to sentiment analysis (using Vader) to provide us with a rounded picture of the types of key words and mix of sentiment used by more successful competitor products. Results from these processes will be compared among the titles with either a high or low comparative rating as indicated by a scaled weighted average of reviews from Goodreads.

The Python script used to conduct this prototype NLP analysis is outlined in [A3_PrototypeNLP](https://github.com/cdshing/WebCrawlerNLP/blob/main/A3_PrototypeNLP.py).

![Imgur Image](https://github.com/cdshing/WebCrawlerNLP/blob/main/T1_Architecture.jpg)

### References:

* Ankit, R. (2021). Python | Sentiment Analysis using VADER. GeeksforGeeks. https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/?ref=lbp


* Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit (1st ed.). O’Reilly Media. https://www.nltk.org/book/


* Burchell, J. (2017). Using VADER to handle sentiment analysis with social media text. T-Redactyl. https://t-redactyl.io/blog/2017/04/using-vader-to-handle-sentiment-analysis-with-social-media-text.html



* Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020). YAKE! Keyword Extraction from Single Documents using Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509, pp 257-289


* Campos, R., Mangaravite, V., Pasquali, A., Jorge, A. M., Nunes, C., & Jatowt, A. (2018). A Text Feature Based Automatic Keyword Extraction Method for Single Documents. Lecture Notes in Computer Science, 684–691. https://doi.org/10.1007/978-3-319-76941-7_63


* Chandra, R. V., & Varanasi, B. S. (2015). Python requests essentials. Packt Publishing Ltd.


* Chaudhary, A. (2020). Unsupervised Keyphrase Extraction. Chaudhary Research & Engineering. https://amitness.com/keyphrase-extraction/


* Demarest, A. (2021). What is Goodreads? Everything you need to know about the popular site for readers and book recommendations. Business Insider Australia. https://www.businessinsider.com.au/what-is-goodreads?r=US&IR=T


* Dumas, N. (2015). GitHub - therealfakemoot/collections2: collections2 is a pure-python reimplementation of several collections objects. GitHub. https://github.com/therealfakemoot/collections2


* Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2


* Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science &amp; Engineering, 9(3), 90–95.


* Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014. https://github.com/cjhutto/vaderSentiment


* Kashif, J. (2020). Python BeautifulSoup Scraper that scrapes book covers, titles, descriptions, average rating, rating and authors from www.goodreads.com. https://gist.github.com/kashiftriffort/a2e7ad06dca5acc38a76007465fe6226


* Katari, K. (2020). Exploratory Data Analysis(EDA): Python - Towards Data Science. Medium. https://towardsdatascience.com/exploratory-data-analysis-eda-python-87178e35b14


* Kooten, P. (2020). GitHub - kootenpv/contractions: Fixes contractions such as `you’re` to you `are`. GitHub. https://github.com/kootenpv/contractions


* Mayo, M. (2020). A General Approach to Preprocessing Text Data. KDnuggets. https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html


* McGinn, B. (2020). The Importance of a Back Cover Book Design. Bailey Designs Books. https://www.baileydesignsbooks.com/back-cover-design/


* Mueller, A. (2020). WordCloud for Python. WordCloud for Python Documentation. https://amueller.github.io/word_cloud/


* Nyar, K. (2018). Python BeautifulSoup Scraper that scrapes book covers, titles and authors from www.goodreads.com. https://gist.github.com/kyawkn/bb2c3f6d5e181a8aade0224553878040


* Pandey, P. (2020). Simplifying Sentiment Analysis using VADER in Python (on Social Media Text). Medium. https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f


* Pedregosa, F., Varoquaux, Ga"el, Gramfort, A., Michel, V., Thirion, B., Grisel, O., … others. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825–2830.


* Rao, P. (2019). Fine-grained Sentiment Analysis in Python (Part 1) - Towards Data Science. Medium. https://towardsdatascience.com/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4


* Reback, J. (2021). pandas-dev/pandas: Pandas 1.3.4. Zenodo. https://zenodo.org/record/5574486


* Regular expression operations — Python 3.10.0 documentation. (2021). Python Documentation. https://docs.python.org/3/library/re.html


* Richardson, L. (2021). Beautiful Soup: We called him Tortoise because he taught us. BeautifulSoup. https://www.crummy.com/software/BeautifulSoup/


* Saving multiple api calls in a json file in Python. (2019). https://stackoverflow.com/questions/54755971/saving-multiple-api-calls-in-a-json-file-in-python


* Sen, O. (2021). Data Scraping Goodreads using Python BeautifulSoup - MLearning.ai. Medium. https://medium.com/mlearning-ai/data-scraping-goodreads-using-python-beautifulsoup-3f4f16960255


* Sevil, C. (2019). It is Legal to Scrape Data From Another Business’ Website? LegalVision. https://legalvision.com.au/scrape-data/


* Shrivastava, I. (2020). Exploring Different Keyword Extractors — Statistical Approaches. Medium. https://medium.com/gumgum-tech/exploring-different-keyword-extractors-statistical-approaches-38580770e282


* Terms of Use. (2021). Goodreads. https://www.goodreads.com/about/terms


* Unidecode. (2021). PyPI. https://pypi.org/project/Unidecode/


* vaderSentiment. (2020). PyPI. https://pypi.org/project/vaderSentiment/


* Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.


* Verma, U. (2020). Text Preprocessing for NLP (Natural Language Processing),Beginners to Master. Retrieved from https://medium.com/analytics-vidhya/text-preprocessing-for-nlp-natural-language-processing-beginners-to-master-fd82dfecf95


* Waskom, M. (2017). Seaborn. Zenodo. https://zenodo.org/record/883859


* WebDriver. (2021). Selenium. https://www.selenium.dev/documentation/webdriver/


* Weng, J. (2021). NLP Text Preprocessing: A Practical Guide and Template. Medium. https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79


* Wille, M. (2021). Amazon’s Goodreads is ancient and terrible. Now there’s an alternative. Input. https://www.inputmag.com/reviews/amazon-goodreads-books-alternative-the-storygraph


* word2number. (2017). PyPI. https://pypi.org/project/word2number/


* yake. (2021). PyPI. https://pypi.org/project/yake/
