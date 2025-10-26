import os  # Importing the 'os' module for operating system related functionalities
import pymongo_conn as mongo_conn
from langchain.chains.question_answering import load_qa_chain  # Importing the function load_qa_chain from langchain library for question answering
import google.generativeai as genai  # Importing the Google Generative AI module from the google package
from langchain_google_genai import ChatGoogleGenerativeAI  # Importing ChatGoogleGenerativeAI class from langchain_google_genai module
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing RecursiveCharacterTextSplitter class from langchain module for text splitting
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Importing GoogleGenerativeAIEmbeddings class from langchain_google_genai module
from langchain_community.vectorstores import FAISS  # Importing FAISS class from langchain module for vector storage
from langchain.prompts import PromptTemplate  # Importing PromptTemplate class from langchain module for prompts
import warnings
from pymongo_conn import MongoDBManager
warnings.filterwarnings('ignore')
api_key =  "-------------------------"
os.environ['GOOGLE_API_KEY'] = api_key
# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)
key = os.environ.get('GOOGLE_API_KEY')




# class for reading the data
# from the MongoDB Database
class ReadData:
    
    # class constructor
    def __init__(self,db=None, collectionname=None):

        self.m = MongoDBManager(db_name=db,collection_name=collectionname)
    
    # reading the collection
    # data as dataframe
    def read_df(self):

        try:
            return self.m.read_collection_as_df()
        except Exception as e:
            raise e
    
    
    def filter_data_as_str(self,news_text_featurename: str,date_time_featurename: str,filterdate: str = None,filtertime: str = None):
        """
        Reads data from MongoDB as a DataFrame, splits date/time into separate columns,
        optionally filters by date or time, and returns a list of dictionaries
        containing text, date, and time.
        """
        try:
            # Read data
            df = self.read_df()

            # Drop _id column if present
            if '_id' in df.columns:
                df = df.drop(['_id'], axis=1)

            # Drop duplicate rows
            df = df.drop_duplicates(keep='last')

            # Split date_time into date and time
            df[['date', 'time']] = df[date_time_featurename].str.split('/', n=1, expand=True)

            # Clean up extra spaces and normalize case
            df['date'] = df['date'].str.strip().str.lower()
            df['time'] = df['time'].str.strip().str.lower()

            # Drop the original date_time column
            df.drop(columns=[date_time_featurename], inplace=True)

            # Apply filters if provided
            if filterdate is not None:
                df = df[df['date'].str.contains(filterdate.strip().lower(), na=False)]
            if filtertime is not None:
                df = df[df['time'].str.contains(filtertime.strip().lower(), na=False)]

            # Remove rows with empty text
            df = df[df[news_text_featurename].str.strip() != '']

            # Convert to list of dicts
            data = df[[news_text_featurename, 'date', 'time']].to_dict(orient='records')

            print(f"Filtered records: {len(data)}")
            return data

        except Exception as e:
            raise e

# declaring a class 
# for google-generative AI API
class ChatGoogleGENAI:
    def __init__(self):
        
        # Initializing the ChatGoogleGenerativeAI object with specified parameters
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Using the 'gemini-pro' model
            temperature=0,  # Setting temperature for generation
            google_api_key=key, # Passing the Google API key
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192, 
        )


class EmbeddingModel:
    def __init__(self, model_name):
        # Initializing GoogleGenerativeAIEmbeddings object with the specified model name
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key)
        
class TextChunks:
    @classmethod
    def get_text_chunks(cls, separator=None, chunksize=None, overlap=None, text=None):
        try:
            # Splitting text into chunks based on specified parameters
            text_splitter = RecursiveCharacterTextSplitter(separators=separator, chunk_size=chunksize, chunk_overlap=overlap)
            return text_splitter.split_text(text)
        except Exception as e:
            raise e
        
class Vectors:
    @classmethod
    def generate_vectors(cls, chunks, model):
        try:
            # Generating vectors from text chunks using specified model
            embeddings = EmbeddingModel(model_name=model).embeddings
            return FAISS.from_texts(chunks, embedding=embeddings)
        except Exception as e:
            raise e

class DocumentSummarization(ChatGoogleGENAI):
    def __init__(self,text: str):
        super().__init__()  # Calling the constructor of the superclass (ChatGoogleGENAI)
        # Reading text from the specified directory and assigning it to self.file
        self.file = text
        

    def get_chunks(self, separator=None, chunksize=None, overlap=None):
        try:
            # Getting text chunks from the file using TextChunks class
            return TextChunks().get_text_chunks(separator=separator, chunksize=chunksize, overlap=overlap, text=self.file)
        except Exception as e:
            raise e
        
    def embeddings(self, separator=None, chunksize=None, overlap=None, model=None):
        try:
            # Generating vectors from text chunks using Vectors class
            return Vectors().generate_vectors(chunks=self.get_chunks(separator, chunksize, overlap), model=model)
        except Exception as e:
            raise e
    
    def summarisation_chains(self, chaintype: str):
        try:
            # Defining a prompt template for summarization
            # the "context" variable provides the background or content from which the summary is derived
            # the "question" variable prompts the user to focus on specific aspects or details within that context
            prompt_template = """
            You are given a text that contains daily stock news for indian stock market. 
            Your job is generate a consice summary of the given text.
            Generate Summary in 1-3 sentence only.
            You have to make sure no information is missed.
            Context:\n {context}?\n
            Question: \n{question}\n
        
            Answer:
            """
            # Creating a PromptTemplate object with the defined template
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            # Loading the question-answering chain with the specified chain type and prompt
            return load_qa_chain(self.model, chain_type=chaintype, prompt=prompt)
        except Exception as e:
            raise e
    
    def main(self, separator=None, chunksize=None, overlap=None, model=None, type=None, user_question=None):
        try:
            # Retrieving the summarization chain
            chain = self.summarisation_chains(chaintype=type)
            # Generating embeddings for the text
            db = self.embeddings(separator, chunksize, overlap, model)
            # Performing similarity search and obtaining documents
            docs = db.similarity_search(user_question)
            # Generating response using the chain and user question
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            return response
        except Exception as e:
            raise e

      
# class for storing
# the summarized text
# into the database
class GenerateTextSummaries:

    def __init__(self, database=None, collection=None, featurename=None, date_time_featurename=None, date_filter=None, time_filter=None):


        r = ReadData(db=database, collectionname=collection)
        self.news_data = r.filter_data_as_str(
            news_text_featurename=featurename,
            date_time_featurename=date_time_featurename,
            filterdate=date_filter,
            filtertime=time_filter
        )

        self.feature = featurename
        self.date_time = date_time_featurename

    def get_summaries(self, chunk=None, overlap=None, model_name=None, chain=None, query=None):
        try:
            summaries = []
            for item in self.news_data:
                text = item[self.feature]
                summarizer = DocumentSummarization(text=text)
                result = summarizer.main(
                    chunksize=chunk,
                    overlap=overlap,
                    model=model_name,
                    type=chain,
                    user_question=query
                )

                summaries.append({
                    'summary': result['output_text']
                })
            return summaries
        except Exception as e:
            raise e

        
if __name__ == '__main__':

    summaries = GenerateTextSummaries(
        database='Vibhor',
        collection='moneycontrol_news',
        featurename='text', date_time_featurename='date_time',
        date_filter='october 21',
        time_filter=None

    )
    result_data = summaries.get_summaries(chunk=1000,overlap=300,model_name='models/gemini-embedding-001',chain='stuff',query='give summary of the following')
    print("\n=== GENERATED SUMMARIES ===")
    for idx, summary in enumerate(result_data, start=1):
        print(f"{idx}. {summary}\n")
