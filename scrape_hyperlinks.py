
import scraping_data as nts


# Creating a driver class to handle the execuetion of the code
# this class execuets the process of scraping and storing the 
# Hyperlinks data
class ScrapeHyperLinkaClass:

    def __init__(self, link=None, filename=None):

        self.links = link
        self.filename = filename
    
    # scraping the links
    def scrapelinks(self, v2=None, v3=None, ref=None):

        try:

            news_link = nts.GetDataFrame(
                link=self.links, val2=v2, val3=v3, ref=ref)
            news_link.createfile(filename=self.filename)
            news_link.releasedriver()

        except Exception as e:
                return e
        
if __name__ == '__main__':
     
    scr_news = ScrapeHyperLinkaClass(
                link="https://www.moneycontrol.com/news/business/stocks/",
                filename='links.csv'
            )

    scr_news.scrapelinks(v2='fleft', v3='a', ref='href')