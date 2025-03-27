# Stock News Scraper with Summarization

## Project Overview

In this project, we are developing a Python-based web scraper that extracts stock market news from the Moneycontrol website using Selenium. The extracted news is then summarized using Google Gemini API and stored in a MongoDB database.

 ***Key Steps Involved:***

  Hyperlink Extraction: A separate Python class is used to scrape and store article hyperlinks before processing the news content.
  
        '''Web Scraping using Selenium python we will use Google Chrome web browser for scraping the text from the website'''

        # creating a class to scrape
        # the hyperlinks from the
        # main page
        class WebScraping:

        def __init__(self,link=None):
            
            # declearing the variable for chrome web driver
    
            self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager("134.0.6998.178").install()))
            self.driver.get(link)

            
        # Creating a driver class to handle the execuetion of the code
        # this class execuets the process of scraping and storing the 
        # Hyperlinks data
        class ScrapeHyperLinkaClass:

        def __init__(self, link=None, filename=None):
    
            self.links = link
            self.filename = filename

    
   Scraping news from Moneycontrol: Using Selenium WebDriver, we extract the news title, date, and content from the website.

        # creating a class
        # to extract data
        # from the stored links
        class ExtractText:

        def __init__(self,link=None):
            
            # declearing the driver variable
            self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager("134.0.6998.178").install()))
            self.driver.get(link)

  Summarizing news using AI: The extracted news content is sent to the Google Gemini API for automatic summarization.

          # class for summarizing
          # the news text
          class DocumentSummarization(ChatGoogleGENAI):
          def __init__(self,text):
              super().__init__()  # Calling the constructor of the superclass (ChatGoogleGENAI)
              # Reading text from the specified directory and assigning it to self.file
              self.file = text

  Storing data in MongoDB: Both the original news content and its summarized version are saved in a MongoDB collection.

         # MongoDB Integration for storing original news text
         class MongoDBManager:

            def __init__(self, db_name=None, collection_name=None):
        
                # Connecting to MongoDB
                self.client = MongoClient("mongodb://localhost:27017/")  # connection string
                self.db = self.client[db_name]
                self.collection = self.db[collection_name]

          # class for storing
          # the summarized text
          # into the database
          class InsertSummariesIntoDatabase:

              def __init__(self,database=None,collection=None,featurename=None,slicer=None):
          
                  r = ReadData(db=database,collectionname=collection)
                  self.news_text = r.filter_data_as_str(news_text_featurename=featurename,slicer=slicer)

## Features Explanation

***This project includes the following main features:***

  1. Web Scraping
     
    We automate web browsing using Selenium WebDriver to extract stock-related news.
  
    The scraper navigates the Moneycontrol website, finds relevant stock market news, and extracts text data.
  
  2. Hyperlink Extraction
     
    Before extracting news content, we collect and store hyperlinks of news articles.
  
    A dedicated class ensures that we reuse the links efficiently without reloading pages multiple times.
  
  3. Summarization using Gemini API
     
    The extracted news is sent to Google Gemini API, which generates a concise summary of the article.
  
    This helps users quickly understand the key points of the news without reading the full article.
  
  4. Data Storage in MongoDB
     
    The extracted news content and summaries are stored in a MongoDB database.

***This allows easy retrieval and further analysis of the data.***



## Tech Stack


| Technology          | Purpose                                      |
|---------------------|----------------------------------------------|
| **Python**         | Main programming language                   |
| **Selenium**       | Automates web scraping from Moneycontrol     |
| **Google Gemini API** | Generates summaries of extracted news   |
| **MongoDB**        | Stores extracted news and summaries         |
| **Chrome WebDriver** | Interacts with the website for scraping  |
| **pymongo**        | Connects Python to MongoDB                  |
| **google-generativeai** | Enables AI-generated summaries       |


        




    


