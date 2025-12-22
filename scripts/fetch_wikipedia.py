import os
import wikipediaapi


wiki_wiki = wikipediaapi.Wikipedia(
    language="en", user_agent="WikiScraper/1.0 (https://github.com/isatyamks/RAG)"
)


def main() -> None:
    page_name = input("Enter the page title to save (e.g. 'India'): ")
    page = wiki_wiki.page(page_name)

    if page.exists():
        text = page.text
        os.makedirs("data/raw", exist_ok=True)
        file_path = f"data/raw/{page_name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Scraped page '{page.title}' and saved to {file_path}")
    else:
        print(f"Page '{page_name}' does not exist.")


if __name__ == "__main__":
    main()
