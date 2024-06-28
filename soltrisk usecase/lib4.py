import pdfplumber

def extract_text_from_pdf(file_path):
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

file_path = 'C:/Users/PAVI/Desktop/flask/soltrisk usecase/static/pdf_folder/iso 27002-2022 copy .pdf'
text = extract_text_from_pdf(file_path)
print(text)

