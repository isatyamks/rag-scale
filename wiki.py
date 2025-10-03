import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='WikiScraper/1.0 (https://github.com/isatyamks/RAG)'
)
page_name = input("")
page = wiki_wiki.page(page_name)  

if page.exists():
    text = page.text  
    with open("wiki_raw.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Scraped page '{page.title}' and saved to wiki_raw.txt")
else:
    print(f"Page '{page.title}' does not exist.")
