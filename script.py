import os
import json
from Bio import Entrez
from openai import OpenAI

# --- Configuration ---
EMAIL = "peter.chu.tango@gmail.com"   # NCBI 要求的 email
# 只搜尋標題含 sulforaphane
SEARCH_TERM = '"sulforaphane"[Title]'
MAX_RESULTS = 30                      # 最終要留下的論文數量
OUTPUT_FILE = "data/research.json"

# --- Services ---
Entrez.email = EMAIL
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Warning] OPENAI_API_KEY 環境變數不存在，將無法產生中文解說與分類。")
client = OpenAI(api_key=api_key) if api_key else None


def analyze_abstract(abstract_en: str) -> dict:
    """
    使用 OpenAI：
    1. 產出「一般人看得懂的中文解說」
    2. 幫論文貼標籤（人類 / 動物 / 疾病 / clinical trial / nutrition-metabolism）
       並註明疾病名稱（如果有）
    回傳一個 dict。
    """
    # 如果沒有可用的 client，就直接回傳 fallback 結果
    if client is None:
        return {
            "explanation_zh": f"（未連線至 OpenAI，以下為英文摘要節錄）\n{abstract_en[:800]}",
            "is_human_study": False,
            "is_animal_study": False,
            "is_disease_related": False,
            "disease_name": "",
            "is_clinical_trial": False,
            "is_nutrition_metabolism": False,
        }

    prompt = f"""
你是一位專業的醫學與營養科普寫作者與文獻分析師。

請閱讀下列英文摘要，然後完成兩件事：
1. 用「大人小孩都看得懂的繁體中文」寫一段白話解說。
   - 不要逐句翻譯，而是用自己的話整理重點。
   - sulforaphane 翻譯成「蘿蔔硫素」
   - 文字分成四段落，用 Bulletpoint: 對健康或疾病預防可能的啟示，研究主題、實驗是怎麼做、主要發現和結果。
   - 不要過度誇大療效、不要下結論說「一定可以治癒」。

2. 根據摘要內容，判斷這篇研究屬於哪些類別（可複選）：
   - 人類研究（human study）
   - 動物研究（animal study）
   - 疾病相關研究（disease-related study）
   - clinical trial（臨床試驗）
   - nutrition / metabolism（營養 / 代謝相關）

如果是「疾病相關研究」，請另外註明主要研究的疾病名稱，例如：
   - "breast cancer"
   - "Alzheimer's disease"
   - "type 2 diabetes"
如果沒有明確疾病，就用空字串 ""。

⚠ 非常重要：
請你「只輸出一個 JSON 物件」，不要有任何多餘文字。
JSON 格式必須長這樣（布林請用 true/false，小寫）：

{{
  "explanation_zh": "這裡是一段給一般人看的中文解說。",
  "is_human_study": true 或 false,
  "is_animal_study": true 或 false,
  "is_disease_related": true 或 false,
  "disease_name": "如果有具體疾病名稱，寫在這裡，否則用空字串",
  "is_clinical_trial": true 或 false,
  "is_nutrition_metabolism": true 或 false
}}

請務必確保：
- 只輸出一個 JSON 物件
- 不要多加註解、說明、文字或程式碼區塊

以下是英文摘要：

{abstract_en}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"[OpenAI / JSON 解析錯誤] {e}")
        # 發生錯誤時，給一個安全的 fallback
        return {
            "explanation_zh": f"（自動產生說明失敗，以下為英文摘要節錄）\n{abstract_en[:800]}",
            "is_human_study": False,
            "is_animal_study": False,
            "is_disease_related": False,
            "disease_name": "",
            "is_clinical_trial": False,
            "is_nutrition_metabolism": False,
        }


def fetch_pubmed_data():
    """
    查詢 PubMed，取得最新 MAX_RESULTS 篇
    「標題包含 sulforaphane」且「有摘要」的論文。
    為確保一定能湊滿 30 篇：一次先抓 50 篇再過濾。
    """
    print(f"Searching PubMed for: '{SEARCH_TERM}'...")

    FETCH_MAX = 50  # 一次多抓一些，之後再篩選

    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=SEARCH_TERM,
            retmax=FETCH_MAX,
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

            # 標題雙重過濾保險：必須包含 sulforaphane
            if "sulforaphane" not in title_en.lower():
                continue

            # --- 摘要 ---
            abstract_list = article_data.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_list, list):
                abstract_en = " ".join([str(t) for t in abstract_list])
            else:
                abstract_en = str(abstract_list)

            if not abstract_en.strip():
                # 沒有摘要就跳過
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

    # 最後只留下最新的 MAX_RESULTS 篇
    papers = papers[:MAX_RESULTS]

    print(f"Fetched {len(papers)} valid articles from PubMed.")
    return papers


def process_and_save_data(raw_data):
    """
    對每篇論文呼叫 OpenAI：
    - 產生中文白話解說
    - 標記類別與疾病名稱
    最後寫入 JSON。
    """
    processed = []
    seen_ids = set()

    for item in raw_data:
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])

        abstract_en = item["abstract_en"]
        analysis = analyze_abstract(abstract_en)

        processed.append({
            "id": item["id"],
            "title_en": item["title_en"],
            "pub_date": item["pub_date"],
            "abstract_en": abstract_en,
            "explanation_zh": analysis.get("explanation_zh", ""),
            "is_human_study": analysis.get("is_human_study", False),
            "is_animal_study": analysis.get("is_animal_study", False),
            "is_disease_related": analysis.get("is_disease_related", False),
            "disease_name": analysis.get("disease_name", ""),
            "is_clinical_trial": analysis.get("is_clinical_trial", False),
            "is_nutrition_metabolism": analysis.get("is_nutrition_metabolism", False),
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

