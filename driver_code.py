import time
import pymongo_conn as mongo_conn
import scraping_data as nts
from text_summarization import InsertSummariesIntoDatabase

# Creating a driver class to handle the execuetion of the code
# this class execuets the process of scraping and storing the data
class DriverClass:

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
    
    # inserting the scraped data into MongoDB's collection
    def insert_to_db(self, feature_name=None, valst=None, start=None, end=None, val=None, val2=None, val3=None, val4=None, titleval=None, tagval=None, endcon=None):

        try:
            files = nts.CreateDocuments(
                file_name=self.filename, feature=feature_name, val_lst=valst, end_con=endcon)
            data = files.get_documents(start=start, end=end, val=val, val2=val2,
                            val3=val3, val4=val4, title_val=titleval, tag_val=tagval)
            return data
           
        except Exception as e:
                return e



if __name__ == '__main__':

    def main(scrape=False):
        if scrape:
            print("Starting scraping process...")
            database_var = mongo_conn.MongoDBManager(db_name='Vibhor', collection_name='moneycontrol_news')
            total_length = database_var.check_collection_length()

            print('Total length: ', total_length)
            start = total_length
            end = start + 2

            print(f'Start at {start} and end till {end}')

            scr_news = DriverClass(
                link="https://www.moneycontrol.com/news/business/stocks/",
                filename='20_sep_links.csv'
            )

            scr_news.scrapelinks(v2='fleft', v3='a', ref='href')

            coll = scr_news.insert_to_db(
                feature_name='Links',
                valst=['/news/business/stock', '/news/business'],
                endcon=True,
                start=start,
                end=end,
                titleval='page_left_wrapper',
                tagval='h1',
                val4='article_schedule',
                val='clearfix',
                val2='content_wrapper',
                val3='p'
            )
            
            print(coll)
            print('Appending values...')

            time.sleep(1)
            database_var.insert_data_in_collection(data=coll)
            database_var.close_conn()
            time.sleep(1)


    def driver_func():
        condition = input('Do you want to scrape (yes/no): ').strip().lower()
        if condition == 'yes':
            main(scrape=True)
        else:
            print("Skipping scraping. Proceeding to summarization...")
            main(scrape=False)


    driver_func()

    db_var = mongo_conn.MongoDBManager(db_name='Vibhor', collection_name='text_summaries')
    total_length = db_var.check_collection_length()
    print(total_length)

    summary = InsertSummariesIntoDatabase(database='Vibhor',collection='moneycontrol_news',featurename='text',slicer=total_length)
    result_data = summary.get_summaries(chunk=1000,overlap=300,model_name='models/embedding-001',chain='stuff',query='give summary of the following')
    print(len(result_data))
    

    print('inserting into database.....')

    
    db_var.insert_data_in_collection(data=result_data)
    db_var.close_conn()

