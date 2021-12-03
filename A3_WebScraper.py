# MA5851: DATA SCIENCE MASTERCLASS 1
# A3: WebCrawler and NLP System
# Task 2: WebCrawler

# Code blocks:
# 1. Import the Required Packages
# 2. WebScraping from Goodreads
# 3. Enrich Data Using the Google Application Programming Interface
# 4. Data Wrangling
# 5. Preliminary Exploratory Data Analysis


##### 1. Import the Required Packages
import pandas as pd
import numpy as np
import re
import time
from selenium import webdriver
import requests
from bs4 import BeautifulSoup as bs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from urllib.parse import quote
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import seaborn as sns


##### 2. WebScraping Goodreads
# Sources:
# - https://gist.github.com/kyawkn/bb2c3f6d5e181a8aade0224553878040
# - https://medium.com/mlearning-ai/data-scraping-goodreads-using-python-beautifulsoup-3f4f16960255
# - https://gist.github.com/kashiftriffort/a2e7ad06dca5acc38a76007465fe6226
# - https://github.com/maria-antoniak/goodreads-scraper/blob/master/get_books.py
# - https://nxrunning.wixsite.com/blog/post/web-scraping-goodreads-with-beautifulsoup-popular-running-books

def get_grim_data(driver, url):
    # Make the Soup
    driver.get(url)
    soup = bs(driver.page_source, 'html.parser')
    titles = soup.find_all('a', class_='bookTitle')
    authors = soup.find_all('a', class_='authorName')
    ratings = soup.find_all('span', attrs={'class': 'greyText smallText'})
    # Build the base DataFrame
    df = pd.DataFrame(columns=['title', 'description', 'author', 'avg_rating', 'rating_count'])
    # Write content to the frame
    for title, author, rating in zip(titles, authors, ratings):
        book_page = requests.get("https://www.goodreads.com" + title["href"])
        book_soup = bs(book_page.content, 'html.parser')
        # Extract titles
        title = title.get_text()
        title = re.sub("[\(\[].*?[\)\]]", "", title)
        # Extract descriptions (where available)
        description_divs = book_soup.find_all("div", {"class": "readable stacked", "id": "description"})
        try:
            description = description_divs[0].find_all("span")[1].text
        except IndexError:
            try:
                description = description_divs[0].find_all("span")[0].text
            except IndexError:
                description = np.nan
        # Extract authors
        author = author.get_text()
        # Extract ratings particulars
        try:
            avg_rating = re.search(r'avg rating ([\d.]+)', rating.text).group(1)
        except:
            avg_rating = 999
        try:
            rating_count = re.search(r'([\d,]+) ratings', rating.text).group(1)
        except:
            rating_count = 1
        # Combine the contents from each title into temp dataframe
        df2 = pd.DataFrame([[title, description, author, avg_rating, rating_count]],
                           columns=['title', 'description', 'author', 'avg_rating', 'rating_count'])
        # Flag in key fields missing as nan
        df2.title.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df2.description.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        # append to fill the dataframe
        df = df.append(df2)
    # Deliver the resulting dataframe
    return df

# firstly we need to sign in to scrape multiple pages
# Source: https://github.com/pipegalera/Goodreads-webscraper
user_email = "charlieshing83@bigpond.com"
password = "Bombadil#83"
driver = webdriver.Chrome('C:/Users/61431/Documents/Dev/chromedriver.exe')
driver.get("https://www.goodreads.com/user/sign_in")
time.sleep(5)
driver.find_element_by_id('user_email').send_keys(user_email)
driver.find_element_by_id('user_password').send_keys(password)
driver.find_elements_by_xpath("//*[@id='emailForm']/form/fieldset/div[5]/input[1]")[0].click()

# secondly we need to loop the function defined above for all active grimdark pages and add results to a dataframe
scraped_df = pd.DataFrame(columns=['title', 'description', 'author', 'avg_rating', 'rating_count'])
for i in range(26):
    i = i+1
    url = "https://www.goodreads.com/shelf/show/grimdark/?page="+str(i)
    scraped_df = scraped_df.append(get_grim_data(driver, url), ignore_index=True)
# convert scraped_df to csv
scraped_df.to_csv('scraped_df.csv', index=False)


##### 3. Enrich Data Using the Google Books API
# https://stackoverflow.com/questions/54755971/saving-multiple-api-calls-in-a-json-file-in-python
# https://www.googleapis.com/books/v1/volumes?q=%22The%20Blade%20Itself%22

# import the goodreads scraped data csv
df = pd.read_csv('scraped_df.csv')
# create a blank dataframe to append results to
supplements = pd.DataFrame(columns=['title',
                                    'api_title',
                                    'api_authors',
                                    'api_publisher',
                                    'api_published_date',
                                    'api_page_count',
                                    'api_print_type',
                                    'api_description',
                                    'api_language'])
# loop requests data from the api
for title in df.title:
    url = "https://www.googleapis.com/books/v1/volumes?q=" + quote(title)
    # request data from the url as a json dict
    data = requests.get(url).json()
    # isolate the required fields from the first (i.e. [0]) search result in each instance
    try:
        api_title = data['items'][0]['volumeInfo']['title']
    except:
        api_title = ''
    try:
        api_authors = data['items'][0]['volumeInfo']['authors']
    except:
        api_authors = ''
    try:
        api_publisher = data['items'][0]['volumeInfo']['publisher']
    except:
        api_publisher = ''
    try:
        api_published_date = data['items'][0]['volumeInfo']['publishedDate']
    except:
        api_published_date = ''
    try:
        api_page_count = data['items'][0]['volumeInfo']['pageCount']
    except:
        api_page_count = ''
    try:
        api_print_type = data['items'][0]['volumeInfo']['printType']
    except:
        api_print_type = ''
    try:
        api_description = data['items'][0]['volumeInfo']['description']
    except:
        api_description = ''
    try:
        api_language = data['items'][0]['volumeInfo']['language']
    except:
        api_language = ''
    # gather new data into numpy array and convert to a pandas dataframe
    title = title
    gateway = np.array([title,
                        api_title,
                        api_authors,
                        api_publisher,
                        api_published_date,
                        api_page_count,
                        api_print_type,
                        api_description,
                        api_language]).reshape(1, 10)
    gateway = pd.DataFrame(gateway, columns=['title',
                                             'api_title',
                                             'api_authors',
                                             'api_publisher',
                                             'api_published_date',
                                             'api_page_count',
                                             'api_print_type',
                                             'api_description',
                                             'api_language'])
    # append this data frame for each unique_titles search produced
    supplements = supplements.append(gateway)

# convert to csv to avoid running the above process repeatedly
supplements.to_csv('Supplements.csv', index=False)


##### 4. Data Wrangling

# import the goodreads scraped data
scraped = pd.read_csv('scraped_df.csv')
# import supplementary google API data
supplements = pd.read_csv('Supplements.csv')
# join with subset dataframe
df = pd.merge(scraped, supplements)

# explore the full list of titles
# print the top 5 titles by frequency
print(pd.DataFrame(df["title"].value_counts(ascending=False)).head())
# Note - There are instances where a title appears more than once
# retain a single instance of each unique title
df = df.drop_duplicates(subset='title', keep='last')

# explore the language each books is written in
print(df["api_language"].value_counts(ascending=False))
# Note - 1,215 out of 1,248 are published in english
# subset to only include the english texts
df = df[df.api_language == 'en']

# explore the description fields
print("Missing goodreads descriptions: " + str(df['description'].isna().sum()))
# replace missing descriptions with the API supplement
df.description.fillna(df.api_description, inplace=True)
print("Remaining missing descriptions: " + str(df['description'].isna().sum()))
# drop the remaining cases of missing description
df = df[df['description'].notna()]

# explore the ratings fields
df['rating_count'] = df['rating_count'].str.replace(',', '').apply(pd.to_numeric)
print("The range for avg_rating is: Min(" + str(df.avg_rating.min()) + ") to Max (" + str(df.avg_rating.max()) + ")")
print("The range for rating_count is: Min(" + str(df.rating_count.min()) + ") to Max (" + str(df.rating_count.max()) + ")")
# Note - there appear to be some extreme outliers (i.e., greater than 1M ratings)
print(df.title[df.rating_count > 1000000])
# Note - these are simply very popular generic fantasy and should be excluded
df = df[(df['rating_count'] > 0) & (df['rating_count'] < 1000000)]
# now we need to derive a weighted average rating for analysis
df["wavg_rating"] = df["avg_rating"] * df["rating_count"]
# but this should really be scaled to a range of 0 to 5 for ease of interpretation
df["wavg_rating"] = minmax_scale(df["wavg_rating"], feature_range=(0, 5), axis=0, copy=True)
# and finally create a flag above or below the median of this scaled rating
df["rating_class"] = np.where(df["wavg_rating"] > df["wavg_rating"].median(), "High", "Low")

# explore the authors each book is written by
print(df["author"].value_counts(ascending=False))
# Note - "Adrian   Collins" only writes grimdark magazines, which are not a competitor for novels
df = df[df.author != "Adrian   Collins"]
# Note - Hiroya Oku writes manga and should be excluded
df = df[df.author != "Hiroya Oku"]

# explore the publishers each book is printed by
print(df["api_publisher"].value_counts(ascending=False))
missing_pubs = df[df.api_publisher.isna()]
# Note - 164 titles has missing publishers
# create a dataframe with all authors and their non-missing publishers
auth_pubs = df[["author", "api_publisher"]]
auth_pubs = auth_pubs[auth_pubs['api_publisher'].notna()]
auth_pubs = auth_pubs.drop_duplicates(subset='author', keep='first')
auth_pubs = auth_pubs.rename(columns={'api_publisher': 'lead_publisher'})
# merge the deduplicated list of authors and their leading publishers
df = df.merge(auth_pubs, how='left', on='author')
# replace missing api_publisher with the merged lead_publisher
df.api_publisher.fillna(df.lead_publisher, inplace=True)
# drop any remaining missing cases
df = df[df['api_publisher'].notna()]

# explore the size of each book by supplementary api_page_count
print("The page count range is: Min(" + str(df.api_page_count.min())
      + ") to Max (" + str(df.api_page_count.max()) + ")")
# Note - there appear to be some outliers in the range
# Let's visualise the distribution before making a decision
plt.plot(df['api_page_count'].value_counts())
plt.xlabel("Number of Pages")
plt.ylabel("Frequency")
plt.show()
# there are certainly outliers, but only those with too few pages (i.e., < 100) need removing
df = df[df.api_page_count > 100]

# explore the length of each titles scraped/supplemented description
df['desc_length'] = df['description'].apply(lambda x: len(x) - x.count(' '))
print("The range for desc_length is: Min("
      + str(df.desc_length.min()) + ") to Max (" + str(df.desc_length.max()) + ")")
# Note - there appear to be some outliers that need removing
plt.hist(x=df.desc_length)
plt.xlabel("Length of Description")
plt.ylabel("Frequency")
plt.show()
# Note - there appears to be descriptions below 50 characters that need removing
df = df[df.desc_length > 50]

# since not all are beneficial to the analysis, we should retain only the required fields
df = df[["title",
         "description",
         "desc_length",
         "author",
         "avg_rating",
         "rating_count",
         "wavg_rating",
         "rating_class",
         "api_publisher",
         "api_page_count"]]

# print the field headings, counts and total frame size
print(df.notna().sum())
print("Total frame size is: " + str(len(df)))

# convert to munged dataframe to csv
df.to_csv('munged_df.csv', index=False)


##### 5. Preliminary Exploratory Data Analysis

# import the harvested munged data
df = pd.read_csv('munged_df.csv')

# Univariate analysis: summary of the numeric (i.e., int and float) fields
print(df[["desc_length", "avg_rating", "rating_count", "wavg_rating", "api_page_count"]].describe())

# Univariate analysis: frequency of publishers
publishers = df["api_publisher"].value_counts(ascending=False)
print(publishers[publishers > 4])

# Bivariate review: scatter plot of the scaled weighted average variable against attributes
df.plot.scatter(x="wavg_rating", y="desc_length").set(ylim=(0, 3000))
plt.savefig('wavg_by_desc.png', bbox_inches='tight')
plt.show()

df.plot.scatter(x="wavg_rating", y="api_page_count").set(ylim=(0, 3000))
plt.savefig('wavg_by_page.png', bbox_inches='tight')
plt.show()

# Bivariate Numeric/Categorical Analysis: high/low performance titles by length
print(df.groupby('rating_class')['desc_length'].median())
sns.boxplot(df.rating_class, df.desc_length).set(ylim=(0, 3000))
plt.savefig('class_by_desc.png', bbox_inches='tight')
plt.show()

print(df.groupby('rating_class')['api_page_count'].median())
sns.boxplot(df.rating_class, df.api_page_count).set(ylim=(0, 3000))
plt.savefig('class_by_page.png', bbox_inches='tight')
plt.show()

# create wordcloud of description


