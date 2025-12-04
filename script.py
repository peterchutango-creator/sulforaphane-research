import json
from Bio import Entrez
from deep_translator import GoogleTranslator

# --- Configuration ---
EMAIL = "peter.chu.tango@gmail.com"
SEARCH_TERM = "Sulforaphane"
MAX_RESULTS = 50
OUTPUT_FILE = "data/research.json"

# --- Services ---
Entrez.email = EMAIL
translator = GoogleTranslator(source="en", target="zh-TW")

### --- 建立 10 歲版本摘要（中文） --- ###
def explain_like_age_10_zh(full_text_zh):
    """
    將完整中文摘要改寫成 10 歲小朋友能懂的版本。
    Rule-based：避免使用生成模型。
    """

    intro = "給 10 歲小朋友的解釋：\n"

    # 把摘要切成句子
    sentences = full_text_zh.replace("。", "。|").split("|")
    short = sentences[:4]  # 只取前 4 句來簡化

    rewritten = []
    for s in short:
        s = s.strip()
        if not s:
            continue

        # 基本易懂替換
        s = s.replace("細胞", "身體裡的小工人")
        s = s.replace("退化", "變舊、變壞")
        s = s.replace("老化", "變得累累、沒力氣")
        s = s.replace("脂質", "油油的東西")
        s = s.replace("累積", "堆積")
        s = s.replace("炎症", "身體裡的紅腫不舒服")
        s = s.replace("代謝", "身體處理東西的能力")

        rewritten.append(s)

    return intro + " ".join(rewritten)


### --- PubMed Data Fetcher --- ###
def fetch_pubmed_data():
    print(f"Searching PubMed for: '{SEARCH_TERM}'...")

    # 搜尋 IDs
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

    # 取得詳細資訊
    try:
        handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        print(f"Error fetching PubMed details: {e}")
        return []

    papers = []

    for article in records.get("PubmedArticle", []):
        try:
            medline_citation = article["MedlineCitation"]
            article_data = medline_citation["Article"]

            pmid = medline_citation["PMID"]
            title_en = str(article_data["ArticleTitle"])

            # --- 取得完整摘要英文 ---
            abstract_list = article_data.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_list, list):
                abstract_en = " ".join([str(t) for t in abstract_list])
            else:
                abstract_en = str(abstract_list)

            if not abstract_en:
                continue

            # 出版日期
            pub_date_data = article_data["Journal"]["JournalIssue"]["PubDate"]
            year = pub_date_data.get("Year", "Unknown")
            month = pub_date_data.get("Month", "Unknown")
            pub_date = f"{year}-{month}"

            papers.append({
                "id": pmid,
                "title_en": title_en,
                "abstract_en": abstract_en,
                "pub_date": pub_date,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        except Exception as e:
            print(f"Skipping article due to parsing error: {e}")
            continue

    return papers


### --- Process + Translate --- ###
def translate_to_zh(text):
    try:
        return translator.translate(text)
    except:
        return "（翻譯失敗）" + text


def process_and_save_data(raw_data):
    processed = []
    seen = set()

    for item in raw_data:
        if item["id"] in seen:
            continue
        seen.add(item["id"])

        # 1️⃣ 完整摘要 → 中文
        abstract_zh = translate_to_zh(item["abstract_en"])

        # 2️⃣ 十歲版本摘要（中文）
        abstract_zh_kid = explain_like_age_10_zh(abstract_zh)

        processed.append({
            "id": item["id"],
            "title_en": item["title_en"],
            "abstract_zh_tw": abstract_zh,
            "abstract_zh_tw_kid": abstract_zh_kid,
            "pub_date": item["pub_date"],
            "pubmed_url": item["pubmed_url"]
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved {len(processed)} records → {OUTPUT_FILE}")


### --- main --- ###
if __name__ == "__main__":
    raw = fetch_pubmed_data()
    if raw:
        process_and_save_data(raw)
