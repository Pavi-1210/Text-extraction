import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import textract

# Set the Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBmCFo6_weCFylhnK85u3HL_NhZfCgxudo"

# Set the Tesseract-OCR path and TESSDATA_PREFIX
tesseract_path = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
tessdata_path = r"C:/Program Files/Tesseract-OCR/tessdata"

os.environ['TESSDATA_PREFIX'] = tessdata_path
os.environ['PATH'] += os.pathsep + os.path.dirname(tesseract_path)

# Paths to the folders
image_folder = "static/images"

# Initialize the ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-pro", max_tokens=500, temperature=0.5)

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "I extracted the text from a PDF but the text is in an unstructured format. Based on the text context, you need to structure and summarize the text concisely:\n\n{text}"
)

# Function to extract text from a PDF file


# Function to extract text from an image file
def extract_text_from_image(image_path):
    return textract.process(image_path, method='tesseract').decode('utf-8')



# Process each image
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # Extract text from the image
    try:
        extracted_text = extract_text_from_image(image_path)
        
        if extracted_text.strip():
            # If text is detected, process it
            print(f"Extracted text from {image_file}:")
            print(extracted_text)
            
            # Chain the prompt and the model together for the extracted text
            chain = prompt | model
            result = chain.invoke({"text": extracted_text})
            
            # Extract the structured content from the result
            structured_content = result.content.strip()
            
            print(f"Structured and summarized data from {image_file}")
            print(structured_content)
            print("---------------------------------------------------------------------")
            print(" ")
        else:
            print(f"No text found in {image_file}.")
    except Exception as e:
        print(f"An error occurred while processing {image_file}: {e}")

