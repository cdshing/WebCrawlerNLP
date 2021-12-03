## A3: WebCrawler and NLP System 
### D1: Overview

In the field of novel marketing, McGinn (2020) describes the back cover blurb or elevator pitch as a combination of taglines and descriptions of characters and central themes that is a genuine deal closer for hard copy and electronic book sales. The purpose of this NLP prototype is to demonstrate a practical solution to sourcing publicly available metadata on novels by sub-genre and identify the building blocks of descriptions used to promote successful (highly rated) products among competitors. The market chosen to prototype is “grimdark”, which is a subgenre of fantasy.

The primary source of public novel data in this process is the Goodreads social cataloguing website, which provides a platform for individuals to track, rate and review books they read. With an emphasis on social, the site represents a community of consumers providing independent feedback just as TripAdvisor does for travel and accommodation.

The secondary data source, for those novels identified by web scraping the Goodreads “grimdark” shelf, is the Google Books API. This represents another publicly available metadata delivery asset that can be queried and extracted to supplement missing data and add additional metadata fields to enrich the NLP analysis planned.

The Python script used to harvest these metadata sources is outlined in [A3_WebScraper](https://github.com/cdshing/WebCrawlerNLP/blob/main/A3_WebScraper.py).

Once the chosen Goodreads fields have been scraped and supplemented with query driven Google Books API fields, the data frames will be merged before undergoing pre-processing data cleansing and finally stored as CVS.

Once the raw data is prepared, the derived “description” field will undergo natural language pre-processing to prepare it for analysis. With the end goal being to assist publishers in creating more successful back page blurbs, key words will then be extracted (using Yake) from the marketing heavy bag of words sourced on each identified title. As shown in the figure below, these identified key words will then be subjected to sentiment analysis (using Vader) to provide us with a rounded picture of the types of key words and mix of sentiment used by more successful competitor products. Results from these processes will be compared among the titles with either a high or low comparative rating as indicated by a scaled weighted average of reviews from Goodreads.

![Imgur Image](https://github.com/cdshing/WebCrawlerNLP/blob/main/T1_Architecture.jpg)

### References:

McGinn, B. (2020). The Importance of a Back Cover Book Design. Bailey Designs Books. https://www.baileydesignsbooks.com/back-cover-design/

vaderSentiment. (2020). PyPI. https://pypi.org/project/vaderSentiment/

yake. (2021). PyPI. https://pypi.org/project/yake/
