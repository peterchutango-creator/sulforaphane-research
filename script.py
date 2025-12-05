import os
import json
from Bio import Entrez
from openai import OpenAI

# --- Configuration ---
EMAIL = "peter.chu.tango@gmail.com"   # NCBI 要求的 email
SEARCH_TERM = '"sulforaphane"[Title]'
MAX_RESULTS = 10                      # ✅ 只抓最新 10 篇
OUTPUT_FILE = "data/research.json"

# --- Services ---
Entrez.email = EMAIL
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def explain_abstract_zh(abstract_en: str) -> str:
    """
    使用 OpenAI 將完整英文 abstract，改寫成一般人可以懂的中文解釋。
    語氣：清楚、生活化、但保持科學精神，不要講幹話。
    """
    prompt = f"""
你是一位擅長用淺顯中文解釋醫學與營養研究的科普寫作者。

請閱讀下面這段英文學術摘要，然後用「一般成年讀者看得懂的繁體中文」做成解說：

要求：
1. 不要逐句翻譯，而是「用自己的話」整理重點。
2. 請說明：
   - 研究在「研究什麼」？（主題與對象）
   - 大概「怎麼做」？（方法，用簡單一句話帶過即可）
   - 找到「什麼結果」？
   - 對一般人有什麼可能意義？（例如：對健康、疾病預防、飲食的啟示）
3. 盡量用生活化的比喻，但不要亂許願或過度保證療效。
4. 字數大約 150～250 字之間。

英文原文摘要如下：
{abstract_en}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[OpenAI Error] {e}")
        # 發生錯誤時，至少保留英文摘要
        return f"（說明生成失敗，以下為原始英文摘要）\n{abstract_en}"

def fetch_pubmed_data():
    """
    查詢 PubMed，取得最新 MAX_RESULTS 篇「標題包含 sulforaphane」且「有摘要」的論文。
    為確保一定能湊滿 10 篇：一次先抓 50 篇再過濾。
    """
    print(f"Searching PubMed for: '{SEARCH_TERM}'...")

    # 一次取多一點，避免遇到沒有摘要的文章不夠數
    FETCH_MAX = 50  

    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=SEARCH_TERM,
            retmax=FETCH_MAX,     # 抓 50 篇（而不是 10）
            sort="pub date",
        )
        record = Entrez.read(handle)
        id_list = record["IdList"]
        handle.close()
    except Exception as e:
        print(f"Error searching PubMed IDs: {e}")
        return []

    if not id_list:
        print("No articles found.")
        return []

    # 取得詳細資料
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

            pmid = str(medline_citation["PMID"])
            title_en = str(article_data["ArticleTitle"]).strip()

            # ❗ 標題必須包含 sulforaphane（雙重確保）
            if "sulforaphane" not in title_en.lower():
                continue

            # --- 摘要 ---
            abstract_list = article_data.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_list, list):
                abstract_en = " ".join([str(t) for t in abstract_list])
            else:
                abstract_en = str(abstract_list)

            # ❗ 沒摘要的文章不收
            if not abstract_en.strip():
                continue

            # --- 發表日期 ---
            pub_date_data = article_data["Journal"]["JournalIssue"]["PubDate"]
            year = pub_date_data.get("Year", "Unknown")
            month = pub_date_data.get("Month", pub_date_data.get("MedlineDate", "Unknown"))
            pub_date = f"{year}-{month}"

            papers.append({
                "id": pmid,
                "title_en": title_en,
                "abstract_en": abstract_en,
                "pub_date": pub_date,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        except Exception as e:
            print(f"Skipping one article due to parsing error: {e}")
            continue

    # 最後只留下最新的 10 篇
    papers = papers[:MAX_RESULTS]

    print(f"Fetched {len(papers)} valid articles from PubMed.")
    return papers


def process_and_save_data(raw_data):
    """對每篇論文呼叫 OpenAI 產生中文白話解釋，並存成 JSON。"""
    processed = []
    seen_ids = set()

    for item in raw_data:
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])

        abstract_en = item["abstract_en"]
        explanation_zh = explain_abstract_zh(abstract_en)

        processed.append({
            "id": item["id"],
            "title_en": item["title_en"],
            "pub_date": item["pub_date"],
            "abstract_en": abstract_en,
            "explanation_zh": explanation_zh,
            "pubmed_url": item["pubmed_url"],
        })

    # 寫入 JSON
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        print(f"Successfully processed and saved {len(processed)} items to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving data to file: {e}")


if __name__ == "__main__":
    raw = fetch_pubmed_data()
    if raw:
        process_and_save_data(raw)
