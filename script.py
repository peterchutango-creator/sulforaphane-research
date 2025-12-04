import json
from Bio import Entrez
# 替換 googletrans，使用 deep-translator
from deep_translator import GoogleTranslator

# --- Configuration ---
# WARNING: 請務必將 YOUR_EMAIL@example.com 替換為您的真實電子郵件地址
#          這是 NCBI/Entrez API 規定的要求。
EMAIL = "peter.chu.tango@gmail.com"  
SEARCH_TERM = "Sulforaphane"
MAX_RESULTS = 50  # 每次獲取的文章數量上限
OUTPUT_FILE = "data/research.json"

# --- Services ---
Entrez.email = EMAIL
# 初始化翻譯器：設定來源語言為英文 ('en')，目標語言為繁體中文 ('zh-TW')
translator = GoogleTranslator(source='en', target='zh-TW') 


def fetch_pubmed_data():
    """(4) 查詢 PubMed API，獲取文章標題和摘要。"""
    print(f"Searching PubMed for: '{SEARCH_TERM}'...")
    
    # 1. 搜尋 IDs
    try:
        handle = Entrez.esearch(db="pubmed", term=SEARCH_TERM, retmax=MAX_RESULTS, sort="pub date")
        record = Entrez.read(handle)
        id_list = record["IdList"]
        handle.close()
    except Exception as e:
        print(f"Error searching PubMed IDs: {e}")
        return []

    if not id_list:
        print("No articles found.")
        return []

    # 2. 獲取詳細資訊 (XML格式)
    try:
        handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        print(f"Error fetching PubMed details: {e}")
        return []
    
    research_items = []
    
    for article in records.get('PubmedArticle', []):
        try:
            medline_citation = article['MedlineCitation']
            article_data = medline_citation['Article']
            
            pmid = medline_citation['PMID']
            title_en = str(article_data['ArticleTitle'])
            
            # 摘要處理
            abstract_list = article_data.get('Abstract', {}).get('AbstractText', [])
            # 確保 abstract_list 是列表，並將其連接成單一字串
            if isinstance(abstract_list, list):
                abstract_en = ' '.join([str(text) for text in abstract_list])
            elif abstract_list:
                abstract_en = str(abstract_list)
            else:
                # 跳過沒有摘要的文章
                continue
            
            # 出版日期 (簡化提取)
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
            print(f"Skipping article {pmid if 'pmid' in locals() else 'unknown'} due to parsing error: {e}") 
            continue
            
    return research_items

def simplify_abstract(text):
    """(5a) 簡化英文摘要到八年級閱讀水平 (使用簡單的句長縮減啟發式)。"""
    sentences = text.split('.')
    # 提取前三個句子作為簡化摘要
    simplified = '. '.join(sentences[:3])
    if len(sentences) > 3 and simplified:
        simplified += '...'
        
    return simplified.strip() if simplified else text

def translate_text(text):
    """(5b) 將簡化的英文文本翻譯成清晰的繁體中文 (zh-tw)。"""
    global translator
    try:
        # 由於翻譯器已初始化為目標語言 'zh-TW'，只需傳入文本即可。
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Translation error for text: {text[:50]}... Error: {e}. Returning English fallback.")
        return f"翻譯失敗 (Translation Failed): {text}"

def process_and_save_data(raw_data):
    """處理、簡化、翻譯數據，然後儲存為 JSON。"""
    processed_data = []
    # 使用 set 來追蹤已處理的文章 ID，避免重複
    processed_ids = set()

    for item in raw_data:
        # 避免重複處理
        if item['id'] in processed_ids:
            continue
        processed_ids.add(item['id'])
        
        # Step 5a: 簡化英文摘要
        item['abstract_simplified_en'] = simplify_abstract(item['abstract_en'])
        
        # Step 5b: 翻譯簡化後的文本至繁體中文
        # 這裡不需指定 target_language，因為 translator 實例已經設定好了
        item['abstract_zh_tw'] = translate_text(item['abstract_simplified_en'])
        
        processed_data.append(item)
        
    # 儲存結構化數據
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # 確保使用 ensure_ascii=False 讓中文正確寫入
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully processed and saved {len(processed_data)} items to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving data to file: {e}")

if __name__ == "__main__":
    raw_data = fetch_pubmed_data()
    if raw_data:
        process_and_save_data(raw_data)




