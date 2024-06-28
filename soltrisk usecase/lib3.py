import cv2
import pytesseract
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

# Set the Google API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Path to the image folder
image_folder = "static/images"

# List all files in the image folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'))]

# Initialize the ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-pro", max_tokens=500, temperature=0.5)

# Create a prompt template with a word limit
prompt = ChatPromptTemplate.from_template(
    "I extracted the text from an image but the text is in an unstructured format. Based on the text context, you need to structure and summarize the text concisely:\n\n{text}"
)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Function to detect tables in an image
def detect_tables(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    return contours

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # Open the image
    image = cv2.imread(image_path)
    
    # Extract general text from the image
    general_text = pytesseract.image_to_string(image)
    
    if general_text.strip():
        # If text is detected, process it
        print(f"Extracted text from {image_file}:")
        print(general_text)
        
        # Chain the prompt and the model together for general text
        chain = prompt | model
        result = chain.invoke({"text": general_text})
        
        # Extract the structured content from the result
        limited_result = result.content.strip()
        
        print(f"Structured and summarized data from {image_file}")
        print(limited_result)
        print("---------------------------------------------------------------------")
        print(" ")
    else:
        # If no text is detected, detect tables in the image
        contours = detect_tables(image)
        if contours:
            # Extract and process each table separately
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                table_image = image[y:y+h, x:x+w]
                table_text = pytesseract.image_to_string(table_image)
                
                print(f"Extracted table {i+1} from {image_file}:")
                print(table_text)
              
                # Chain the prompt and the model together for table text
                chain = prompt | model
                result = chain.invoke({"text": table_text})
                
                # Extract the structured content from the result
                limited_result = result.content.strip()
                
                print(f"Structured and summarized data from table {i+1} in {image_file}")
                print(limited_result)
                print("---------------------------------------------------------------------")
                print(" ")
                
                # Optional: save the cropped table image
                cv2.imwrite(f'table_{i+1}.png', table_image)
                
                # Optional: draw bounding boxes around detected tables
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break  # Exit the loop after processing the first table
        else:
            print(f"No text or tables found in {image_file}.")
    
    # Optional: display the image with bounding boxes
    # cv2.imshow('Processed Image', image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    # break  # Exit the loop after processing the first image

