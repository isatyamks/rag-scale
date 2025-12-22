import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='RAG-Scale/1.0 (https://github.com/isatyamks/RAG)'
)

page = wiki.page("Delhi")

print("=== SUMMARY ===")
print(page.summary[:500])
print("\n\n=== TEXT (first 1000 chars) ===")
print(page.text[:1000])
print("\n\n=== TEXT LENGTH ===")
print(f"Total length: {len(page.text)} characters")
