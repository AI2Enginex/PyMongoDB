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
    def __init__(self,db: str, collectionname: str):

        self.m = MongoDBManager(db_name=db,collection_name=collectionname)
    
    # reading the collection
    # data as dataframe
    def read_df(self):

        try:
            return self.m.read_collection_as_df()
        except Exception as e:
            raise e
    
    # filtering the dataframe to get the summaries
    def filter_data_as_str(self,filter: str, featurename: str, value: str):

        try:
            df = self.read_df()
            df = df[df[filter].str.contains(value, case=False, regex=False)]
            if len(df.index) > 0:
                return df[featurename].to_list()
            else:
                return None
        except Exception as e:
            raise e


# declaring a class 
# for google-generative AI API
class ChatGoogleGENAI:
    def __init__(self):
        
        # Initializing the ChatGoogleGenerativeAI object with specified parameters
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Using the 'gemini-pro' model
            temperature=0.3,  # Setting temperature for generation
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
    def get_text_chunks(cls, separator: None, chunksize: str, overlap: str, text: str):
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

# =================== GEMINI PARAMETERS CONFIG CLASS ===================

class GeminiParameters:
    """
    Holds all configuration parameters related to Gemini summarization and embeddings.
    """
    def __init__(
        self,
        separator: str = None,
        chunk_size: int = 3000,
        overlap: int = 100,
        model_name: str = "models/gemini-embedding-001",
        chain_type: str = "stuff",
        query: str = "Summarize the following text"
    ):
        self.separator = separator
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model_name = model_name
        self.chain_type = chain_type
        self.query = query


# =================== MAIN SUMMARIZATION CLASS ===================

class DocumentSummarization(ChatGoogleGENAI):
    def __init__(self, text: str, gemini_params: GeminiParameters):
        """
        text: input text to summarize
        gemini_params: instance of GeminiParameters with all configuration
        """
        super().__init__()
        self.file = text
        self.params = gemini_params

    def get_chunks(self):
        try:
            # Getting text chunks using parameters from GeminiParameters
            return TextChunks().get_text_chunks(
                separator=self.params.separator,
                chunksize=self.params.chunk_size,
                overlap=self.params.overlap,
                text=self.file
            )
        except Exception as e:
            raise e
        
    def embeddings(self):
        try:
            # Generating vectors from text chunks using the Gemini embedding model
            return Vectors().generate_vectors(
                chunks=self.get_chunks(),
                model=self.params.model_name
            )
        except Exception as e:
            raise e
    
    def summarisation_chains(self):
        try:
            # Defining the prompt template for summarization
            prompt_template = """
            You are given a text that contains daily stock news for the Indian stock market.
            Your job is to generate a concise summary of the given text.
            Generate Summary in 1-2 sentences only.
            You have to make sure no information is missed.

            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
            """

            # Creating a PromptTemplate object with the defined template
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Loading the question-answering chain with the specified chain type and prompt
            return load_qa_chain(
                self.model,
                chain_type=self.params.chain_type,
                prompt=prompt
            )
        except Exception as e:
            raise e
    
    def main(self):
        try:
            # Retrieve summarization chain
            chain = self.summarisation_chains()

            # Generate embeddings
            db = self.embeddings()

            # Perform similarity search
            docs = db.similarity_search(self.params.query)

            # Generate the final summarized response
            response = chain(
                {"input_documents": docs, "question": self.params.query},
                return_only_outputs=True
            )
            return response
        except Exception as e:
            raise e


# =================== BULK SUMMARIZATION CLASS ===================

class GenerateTextSummaries:
    def __init__(self, database: str, collection: str, filter_col: str, featurename: str, val: str):
        r = ReadData(db=database, collectionname=collection)
        self.news_text = r.filter_data_as_str(
            filter=filter_col,
            featurename=featurename,
            value=val
        )
    
    def clean_data(self):
        try:
            return [data for data in self.news_text if len(data.strip()) > 1]
        except Exception as e:
            raise e

    def create_list_of_summaries(self, data: list):
        try:
            return [text for text in data]
        except Exception as e:
            raise e

    def get_summaries(self, gemini_params: GeminiParameters):
        try:
            summaries = []
            for data in self.clean_data():
                re = DocumentSummarization(text=data, gemini_params=gemini_params)
                result = re.main()
                summaries.append(result['output_text'])

            return self.create_list_of_summaries(data=summaries)
        except Exception as e:
            raise e


# =================== MAIN EXECUTION ===================

if __name__ == '__main__':
    #  Create Gemini parameter configuration
    gemini_params = GeminiParameters(
        separator=None,
        chunk_size=3000,
        overlap=100,
        model_name="models/gemini-embedding-001",
        chain_type="stuff",
        query="summarize the following text"
    )

    # Initialize DB summary generator
    text_summaries = GenerateTextSummaries(
        database='Vibhor',
        collection='moneycontrol_news',
        filter_col='date_time',
        featurename='text',
        val='october 25'
    )
    
    # Generate summaries using the Gemini config
    summaries = text_summaries.get_summaries(gemini_params=gemini_params)

    #Print all summaries
    print("\n=== GENERATED SUMMARIES ===")
    for idx, summary in enumerate(summaries, start=1):
        print(f"{idx}. {summary}\n")