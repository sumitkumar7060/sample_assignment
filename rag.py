import streamlit as st
import fitz  # PyMuPDF for PDF parsing
import cohere
import fitz  # PyMuPDF for PDF handling
import tiktoken  # for tokenization
import pytesseract
from PIL import Image
import io
import chromadb
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import re

load_dotenv()
client = AzureOpenAI(  
api_version="2024-02-01"
)

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

class Utils:
    def __init__(self):
        self.client=self.clientActivator()
        
    def clientActivator(self):
        try:
            return chromadb.HttpClient(host='localhost', port=1998)
        except Exception as e:
            print("VECTOR DATABASE CONNECTION FAILED")
            print(e)
    
    # def pdfReaderWithOCR(self,filePath):
    #     text = ""
    #     with fitz.open(filePath) as doc:
    #         count=0
    #         for page in doc:
    #             count+=1
    #             print(count)
    #             text += page.get_text()
    #             images = page.get_images(full=True)
    #             for img_index, img in enumerate(images):
    #                 xref = img[0]
    #                 # print(xref)
    #                 base_image = doc.extract_image(xref)
    #                 # print(base_image)
    #                 image_bytes = base_image["image"]
    #                 # print(img_index," ",image_bytes)
    #                 pil_image = Image.open(io.BytesIO(image_bytes))
    #                 pil_image = pil_image.convert("RGB")
    #                 text += pytesseract.image_to_string(pil_image)
    #     return text

        
    
    def textToChunks(self,text, encoding_name="cl100k_base"):
        text = text.lower()
        # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        chunk_size = 500  # Maximum tokens per chunk
        overlap = 0  # Token overlap between chunks

        chunks = []
        start_idx = 0
        end_idx = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunks.append(tokens[start_idx:end_idx])
            start_idx += chunk_size - overlap

        # Convert token lists back to strings
        chunk_texts = [''.join(encoding.decode(chunk)) for chunk in chunks]
        return chunk_texts
    
    def createEmbeddings(self,texts,collectionName):
        if self.client is None and not self.client.heartbeat() :
            self.client=self.clientActivator()
        collection = self.client.get_or_create_collection(name=collectionName,metadata={"hnsw:space": "cosine"})
        collection.delete(
            where={"sourceid":collectionName},
        )
        ids=[f"{collectionName}{i}" for i in range(len(texts))]
        metadata=[{"sourceid":collectionName} for i in range(len(texts))]    # 
        print(len(texts))
        print(len(ids))
        collection.add(documents=texts,ids=ids,metadatas=metadata)

    def getEmbeddings(self,message,collectionName):
        if self.client is None and not self.client.heartbeat() :
            self.client=self.clientActivator()
        collection = self.client.get_or_create_collection(name=collectionName,metadata={"hnsw:space": "cosine"})
        if collectionName is None:
            return collection.query(
            query_texts=[message],
            n_results=5,
        )["documents"][0]

        # print(collection.count())
        return collection.query(
            query_texts=[message],
            n_results=5,
            where={"sourceid":collectionName},
            # where_document={"$contains":"search_string"}
        )["documents"][0]
        

    def check_collection(self,collectionName):
         if self.client is None and not self.client.heartbeat() :
            self.client=self.clientActivator()
            collections = self.client.list_collections()
            collection_exists = any(col.name == collectionName for col in collections)
            if collection_exists:
                return True
            else:
                return False



    def deleteEmbeddings(self,collectionName):
        if self.client is None and not self.client.heartbeat() :
            self.client=self.clientActivator()
        collection = self.client.get_or_create_collection(name=collectionName,metadata={"hnsw:space": "cosine"})
        
        collection.delete(
            where={"sourceid":collectionName},
        )
        if collection.count()==0:
            self.client.delete_collection(collectionName)
        print("deleted successfully")
         
    def processPdf(self,text,collectionName):
        try:
            result=self.textToChunks(text)
            self.createEmbeddings(collectionName=collectionName,texts=result)
            return True
        except Exception as e:
            return False


# Load models
st.title("Document-based Q&A Bot with RAG")
st.write("Upload a document and ask a question. The bot will retrieve the relevant information from the document and answer your question.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text




def runFile(context,message):
            message=[
                {"role": "system", "content": f'''YOU ARE AN HELPFUL ASSISTANT. YOUR TASK IS TO USE YOUR EXPERTISE TO PROVIDE THE MOST ACCURATE, DETAILED, AND ACTIONABLE ADVICE OR SOLUTION BASED ON THE FOLLOWING CONTEXT:
                ###INSTRUCTIONS###
                - THOROUGHLY ANALYZE THE GIVEN CONTEXT.
                - APPLY YOUR DOMAIN KNOWLEDGE TO GENERATE A HIGH-QUALITY RESPONSE.
                - ENSURE CLARITY, PRECISION, AND PRACTICALITY IN YOUR ANSWER.
                
                ###CONTEXT###
                {context}
                
                ###WHAT NOT TO DO###
                - NEVER PROVIDE GENERIC OR VAGUE RESPONSES.
                - NEVER IGNORE THE DETAILS OF THE CONTEXT PROVIDED.
                - NEVER STRAY FROM YOUR EXPERTISE.'''},
                {"role": "user", "content": message},
            ]
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=message
            )
            return response.choices[0].message.content

# Upload document
uploaded_file = st.file_uploader("Upload your document (PDF)", type=["pdf"])
if uploaded_file:
    filename = uploaded_file.name
    short_filename = re.sub(r'[^a-zA-Z0-9]', '', filename)

if uploaded_file is not None:
    try:
        st.write("Document uploaded successfully!")
        obj = Utils()
        
        question = st.text_input("Ask a question about the document:")
        res=obj.check_collection(collectionName=short_filename[:45])
        if res:        
            context = ' '.join(obj.getEmbeddings(message=question, collectionName=short_filename[:45]))
        else:
            document_text = extract_text_from_pdf(uploaded_file)
            print("doc : ",document_text)
            resut=obj.processPdf( document_text,collectionName=short_filename[:45])
            context = ' '.join(obj.getEmbeddings(message=question, collectionName=short_filename[:45]))

        response=runFile(context,question)

        if response:
            st.write(f"Answer: {response}")
        else:
            st.write("No answer found in the response.")
    except Exception as e:
        st.write(f"Error: {e}")


