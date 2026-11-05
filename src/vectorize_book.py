import os
# For loading environment variable
from dotenv import load_dotenv

#import UnstructuredFileLoader to load and extract the content from individual files
#import DirectoryLoader to load the files from a specified directory
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
# import CharacterTextSplitter to split the text into chunk
from langchain_text_splitters import CharacterTextSplitter
# import HuggingFaceEmbeddings to convert chunk into vectors
from langchain_huggingface import HuggingFaceEmbeddings
# import chroma db to store vector in database
from langchain_chroma import Chroma

load_dotenv
CLASS_SUBJECT_NAME = os.getenv('CLASS_SUBJECT_NAME')
DEVICE = os.getenv('DEVICE','CPU') #Default to 'cpu' if not set
working_dir = os.path.dirname(os.path.abspath(__file__))
print("working_dir:",working_dir)
parent_dir = os.path.dirname(working_dir)
print("parent_dir:",parent_dir)
data_dir = f"{parent_dir}/data"
vector_db_dir = f"{parent_dir}/vector_db"
chapters_vector_db_dir = f"{parent_dir}/chapters_vector_db"
print("data_dir:",data_dir)
print("vector_db_dir:",vector_db_dir)
print("chapters_vector_db_dir:",chapters_vector_db_dir)

# initialize the embedding model
embedding = HuggingFaceEmbeddings() # use Device from  env

# initialize textsplitter
text_splitter = CharacterTextSplitter(chunk_size=2000,chunk_overlap=500)

def vectorize_book_and_store_to_db(class_subject_name,vector_db_name):
    #Define the function that takes CLASS_SUBJECT_NAME and vector databse name as input

    book_dir = f"{data_dir}/{class_subject_name}"
    #Create the complete path of the directory containing subject's Pdf book

    vector_db_path = f"{vector_db_dir}/{vector_db_name}"
    #Create the complete path where vector database will be stored

    loader = DirectoryLoader(
        path=book_dir,
        glob="./*.pdf",
        loader_cls=UnstructuredFileLoader
    )
    # initialize DirectoryLoader to find the all pdf files from specified directory
    # Use UnstructuredFileLoader to extract the text from the Pdf files
    # glob ="./*.pdf" ensures that only pdf files are selected
    documents = loader.load()
    #Load all pdf files and extract their content into document objects

    text_chunks = text_splitter.split_documents(documents)
    # split extracted document into smaller text chunks
    # This makes documents easier to process and retreive

    Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        persist_directory=vector_db_path
    )

    # convert text chunks into vector using embedding model
    # store generated vectors and associated document data into chroma db
    #persist directory specifies location where the vector database is saved

    print(f"{class_subject_name} saved to vector db:{vector_db_name}")
      # Display a confirmation message after storing the subject's data.

    return 0
    # Return 0 to indicate that the function has completed successfully.

def vectorize_chapters(class_subject_name):
    book_dir = f"{data_dir}/{class_subject_name}"
    #  create the book directory with subject name
    for chapter in os.listdir(book_dir):
        #iterate  all the chapter from book directory
        if not chapter.endswith('.pdf'):
            # if chapter does not start with ".pdf", we will skip it
            continue

        chapter_name = chapter[::-4]
        # we will remove the .pdf from chapter name
        chapter_pdf_path = f"{book_dir}/{chapter}"
        # Create the complete file path of the current chapter PDF.
        loader = UnstructuredFileLoader(chapter_pdf_path)
        # Initialize UnstructuredFileLoader to load the current chapter PDF.
        documents = loader.load()
        # Load the PDF and extract its text into document objects.
        texts = text_splitter.split_documents(documents)
        # Split the extracted chapter content into smaller text chunks.
        Chroma.from_documents(documents=texts, 
                                  embedding=embedding, 
                                  persist_directory=f"{chapters_vector_db_dir}/{chapter_name}")
        # Convert the chapter's text chunks into vector embeddings.
        # Store the embeddings in a separate Chroma vector database.
        # Each chapter gets its own database directory using its chapter name.
        print(f"{chapter_name} chapter vectorized")
             # Display a confirmation message after vectorizing the chapter.
    return 0
    # Return 0 to indicate that the function has completed successfully.
