import json
from Bio import Entrez
from googletrans import Translator

# --- Configuration ---
EMAIL = "YOUR_EMAIL@example.com"  # REQUIRED by NCBI/Entrez
SEARCH_TERM = "Sulforaphane"
MAX_RESULTS = 50  # Number of articles to fetch
OUTPUT_FILE = "data/research.json"

# --- Services ---
Entrez.email = EMAIL
# NOTE: The googletrans library is unofficial and may fail. 
# For production, use a formal service (e.g., Google Cloud Translation API)
try:
    translator = Translator()
except Exception as e:
    print(f"Warning: Failed to initialize Google Translator: {e}")
    translator = None


def fetch_pubmed_data():
    """Fetches PubMed IDs and then details for the search term."""
    print(f"Searching PubMed for: '{SEARCH_TERM}'...")

    # 1. Search for IDs
    handle = Entrez.esearch(db="pubmed", term=SEARCH_TERM, retmax=MAX_RESULTS, sort="pub date")
    record = Entrez.read(handle)
    id_list = record["IdList"]
    handle.close()

    if not id_list:
        print("No articles found.")
        return []

    # 2. Fetch details (titles, abstracts) for the IDs
    # Retrieving XML allows for more structured parsing
    handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    research_items = []

    for article in records['PubmedArticle']:
        try:
            medline_citation = article['MedlineCitation']
            article_data = medline_citation['Article']

            # Extract basic data
            pmid = medline_citation['PMID']
            title_en = str(article_data['ArticleTitle'])
            abstract_list = article_data['Abstract']['AbstractText']
            # Join abstract text parts into a single string
            abstract_en = ' '.join([str(text) for text in abstract_list])

            # Publication Date (simplified retrieval)
            pub_date_data = article_data['Journal']['JournalIssue']['PubDate']
            year = pub_date_data.get('Year', 'Unknown')
            month = pub_date_data.get('Month', 'Unknown')
            pub_date = f"{year}-{month}"

            research_items.append({
                "id": pmid,
                "title_en": title_en,
                "abstract_en": abstract_en,
                "pub_date": pub_date,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        except Exception as e:
            # print(f"Skipping article due to error: {e}") 
            continue

    return research_items

def simplify_abstract(text):
    """(5a) Simplifies English abstract to an 8th-grade reading level.
    (Placeholder: Uses a simple sentence shortening heuristic)
    """
    sentences = text.split('.')
    # Take the first three sentences for a simplified summary
    simplified = '. '.join(sentences[:3])
    if len(sentences) > 3 and simplified:
        simplified += '...'

    return simplified.strip() if simplified else text

def translate_text(text, target_language='zh-tw'):
    """(5b) Translates the simplified text into Traditional Chinese."""
    if not translator:
        return "Translation service unavailable."
    try:
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        print(f"Translation error for text: {text[:50]}... Error: {e}. Returning English fallback.")
        return f"Translation Failed: {text}"

def process_and_save_data(raw_data):
    """Processes, simplifies, and translates the data, then saves to JSON."""
    processed_data = []
    for item in raw_data:
        # Step 5a: Simplify English abstract
        item['abstract_simplified_en'] = simplify_abstract(item['abstract_en'])

        # Step 5b: Translate simplified text to Traditional Chinese (zh-tw)
        item['abstract_zh_tw'] = translate_text(item['abstract_simplified_en'], target_language='zh-tw')

        processed_data.append(item)

    # Save the structured data
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully processed and saved {len(processed_data)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    raw_data = fetch_pubmed_data()
    if raw_data:
        process_and_save_data(raw_data)
