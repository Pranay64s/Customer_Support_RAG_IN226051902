import PyPDF2

with open('pdf_text.txt', 'w', encoding='utf-8') as out_file:
    reader = PyPDF2.PdfReader('RAG_Project_Complete.docx.pdf')
    text = ''
    for i in range(len(reader.pages)):
        text += reader.pages[i].extract_text() + '\n'
    out_file.write(text)
