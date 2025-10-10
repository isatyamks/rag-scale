
import wikipediaapi
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data import get_creds, upload_txt
from googleapiclient.discovery import build

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='WikiScraper/1.0 (https://github.com/isatyamks/RAG)'
)
page_name = input("Enter the Page Name: ")



page_list = [   ]

for page_name in page_list:  


    page = wiki_wiki.page(page_name)

    if page.exists():
        text = page.text
        os.makedirs("data/raw", exist_ok=True)
        file_path = f"data/raw/{page_name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Scraped page '{page.title}' and saved to {file_path}")

        creds = get_creds()
        service = build("drive", "v3", credentials=creds)
        FOLDER_ID = "1FDgRBw38w10oT2r9EAD3L9GlU-Hj2sMV"
        upload_txt(service, file_path, FOLDER_ID)
    else:
        print(f"Page '{page.title}' does not exist.")
