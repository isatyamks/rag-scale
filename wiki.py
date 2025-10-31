import wikipediaapi
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

wiki_wiki = wikipediaapi.Wikipedia(
    language="en", user_agent="WikiScraper/1.0 (https://github.com/isatyamks/RAG)"
)

# page_titles = [
#     "India", "Delhi", "New Delhi", "Artificial Intelligence", "Machine Learning",
#     "Deep Learning", "Natural Language Processing", "Computer Vision", "TensorFlow", "PyTorch",
#     "United States", "China", "Japan", "Germany", "France",
#     "Brazil", "Russia", "Canada", "Australia", "Mexico",
#     "United Kingdom", "South Korea", "Italy", "Spain", "South Africa",
#     "India-Pakistan Relations", "World War II", "Quantum Computing", "Blockchain", "Climate Change",
#     "COVID-19 Pandemic", "Mars", "Black Hole", "Neutron Star", "Artificial Neural Network",
#     "Data Science", "Big Data", "Internet of Things", "Smartphone", "Electric Vehicle",
#     "SpaceX", "Tesla", "Amazon", "Microsoft", "Google",
#     "Facebook", "Twitter", "Instagram", "LinkedIn", "TikTok",
#     "Python (programming language)", "JavaScript", "Java", "C++", "Ruby",
#     "HTML", "CSS", "SQL", "PHP", "Swift",
#     "Machine Learning Algorithms", "Data Structures", "Algorithms", "Operating System", "Computer Network"
# ]

page_titles = [
    "Tourism_in_India",
    "Tourism_in_India_by_state",
    "Ministry_of_Tourism_(India)",
    "Tourism",
    "India_Tourism_Development_Corporation",
    "Incredible_India",
]


for page_name in page_titles:

    page = wiki_wiki.page(page_name)

    if page.exists():
        text = page.text
        os.makedirs("data/raw", exist_ok=True)
        file_path = f"data/raw/{page_name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Scraped page '{page.title}' and saved to {file_path}")
    else:
        print(f"Page '{page.title}' does not exist.")
