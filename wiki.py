import wikipediaapi
import os

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='WikiScraper/1.0 (https://github.com/isatyamks/RAG)'
)
page_name = input("Enter the Page Name: ")
page = wiki_wiki.page(page_name)  

if page.exists():
    text = page.text  
    
    # Ensure the data/raw directory exists
    os.makedirs("data\\raw", exist_ok=True)
    
    with open(f"data\\raw\\{page_name}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Scraped page '{page.title}' and saved to data\\raw\\{page_name}.txt")
else:
    print(f"Page '{page.title}' does not exist.")
