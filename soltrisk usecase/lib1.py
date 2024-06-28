import os
from pdfminer.high_level import extract_text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Set the Google API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Path to the PDF folder
pdf_folder = "static/pdf_folder"

# List all files in the PDF folder
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('pdf')]

# Initialize the ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-pro", max_tokens=500, temperature=0.5)

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "I extracted the text from a PDF but the text is in an unstructured format. Based on the text context, you need to structure and summarize the text concisely:\n\n{text}"
)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# Process each PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    
    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if extracted_text.strip():
        # If text is detected, process it
        print(f"Extracted text from {pdf_file}:")
        print(extracted_text)
        
        # Chain the prompt and the model together for the extracted text
        chain = prompt | model
        result = chain.invoke({"text": extracted_text})
        
        # Extract the structured content from the result
        structured_content = result.content.strip()
        
        print(f"Structured and summarized data from {pdf_file}")
        print(structured_content)
        print("---------------------------------------------------------------------")
        print(" ")
    else:
        print(f"No text found in {pdf_file}.")

