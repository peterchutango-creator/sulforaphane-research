import os
import json
from typing import List, Dict, Any
from Bio import Entrez
from openai import OpenAI

# --- Configuration ---
EMAIL = "peter.chu.tango@gmail.com"   # NCBI 要求的 email
SEARCH_TERM = '"sulforaphane"[Title]' # 只搜尋標題含 sulforaphane
MAX_RESULTS = 30                      # 最終要留下的論文數量
OUTPUT_FILE = "data/research.json"

# --- Services ---
Entrez.email = EMAIL
# 如果你有 NCBI API key，可以解開這行
# Entrez.api_key = os.getenv("NCBI_API_KEY")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Warning] OPENAI_API_KEY 環境變數不存在，將無法產生中文解說與分類。")
client = OpenAI(api_key=api_key) if api_key else None


def clean_json_str(content: str) -> str:
    """
    有時模型會回傳 ```json ... ``` 的格式，
    這裡先把外層 code block 剝掉再丟給 json.loads。
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def analyze_abstract(abstract_en: str) -> Dict[str, Any]:
    """
    使用 OpenAI：
    1. 產出「大人小孩看得懂的中文解說」，分成 para1–para4 四段
    2. 幫論文貼標籤（人類 / 動物 / 疾病 / clinical trial）
    回傳一個 dict。
    """
    # 如果沒有可用的 client，就直接回傳 fallback 結果
    if client is None:
        fallback_text = f"（未連線至 OpenAI，以下為英文摘要節錄）\n{abstract_en[:800]}"
        return {
            "para1": fallback_text,
            "para2": "",
            "para3": "",
            "para4": "",
            "is_human_study": False,
            "is_animal_study": False,
            "is_disease_related": False,
            "disease_name": "",
            "is_clinical_trial": False,
        }

    prompt = f"""
你是一位專業的醫學與營養科普寫作者與文獻分析師。

請閱讀下列英文摘要，然後完成兩個任務：

1. 用「大人小孩都看得懂的繁體中文」寫白話解說，拆成四個段落：
   - para1：對健康或疾病預防可能的啟示
   - para2：研究主題是什麼（研究在關心什麼問題）
   - para3：實驗是怎麼做的（對象、方法、大方向）
   - para4：主要發現和結果（但不要說一定治癒、一定有效）
   每一段大約 200–250 字。
   注意：
   - 不要逐句翻譯，而是用自己的話整理重點。
   - sulforaphane 翻譯成「蘿蔔硫素」
   - broccoli 翻譯成「青花椰菜」
   - 不要過度誇大療效、不要下結論說「一定可以治癒」。

2. 根據摘要內容，判斷這篇研究屬於哪些類別（可複選），並用布林值標示：
   - is_human_study：人類研究（human study）
   - is_animal_study：動物研究（animal study）
   - is_disease_related：是否為疾病相關研究（disease-related study）
   - is_clinical_trial：是否為 clinical trial（臨床試驗）

如果是「疾病相關研究」，請另外註明主要研究的疾病中文和英文名稱，例如：
   - "乳癌 Breast Cancer"
   - "阿茲罕默症 Alzheimer's Disease"
   - "高血壓 Hypertension"
如果沒有明確疾病，就用空字串 ""。

⚠ 非常重要：
請你「只輸出一個 JSON 物件」，不要有任何多餘文字。
JSON 格式必須長這樣（布林請用 true/false，小寫）：

{{
  "para1": "第一段：對健康或疾病預防可能的啟示。",
  "para2": "第二段：研究主題。",
  "para3": "第三段：實驗是怎麼做。",
  "para4": "第四段：主要發現和結果。",
  "is_human_study": true 或 false,
  "is_animal_study": true 或 false,
  "is_disease_related": true 或 false,
  "disease_name": "如果有具體疾病名稱，寫在這裡，否則用空字串",
  "is_clinical_trial": true 或 false
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
            max_tokens=1400,
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        content = clean_json_str(content)
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"[OpenAI / JSON 解析錯誤] {e}")
        # 發生錯誤時，給一個安全的 fallback
        fallback_text = f"（自動產生說明失敗，以下為英文摘要節錄）\n{abstract_en[:800]}"
        return {
            "para1": fallback_text,
            "para2": "",
            "para3": "",
            "para4": "",
            "is_human_study": False,
            "is_animal_study": False,
            "is_disease_related": False,
            "disease_name": "",
            "is_clinical_trial": False,
        }


def fetch_pubmed_data() -> List[Dict[str, Any]]:
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
            sort="pub date",  # 已經是由新到舊排序
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

    papers: List[Dict[str, Any]] = []

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

            # --- 發表日期（純文字，用於前端顯示） ---
            pub_date_data = article_data["Journal"]["JournalIssue"]["PubDate"]
            year = pub_date_data.get("Year")
            month = pub_date_data.get("Month")
            medline = pub_date_data.get("MedlineDate")

            if year and month:
                pub_date = f"{year} {month}"
            elif medline:
                pub_date = medline
            elif year:
                pub_date = year
            else:
                pub_date = "Unknown"

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


def process_and_save_data(raw_data: List[Dict[str, Any]]) -> None:
    """
    對每篇論文呼叫 OpenAI：
    - 產生四段中文白話解說（para1~para4）
    - 組合 explanation_zh（方便前端一次顯示）
    - 標記類別與疾病名稱
    最後寫入 JSON。
    """
    processed: List[Dict[str, Any]] = []
    seen_ids = set()

    for item in raw_data:
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])

        abstract_en = item["abstract_en"]
        analysis = analyze_abstract(abstract_en)

        para1 = analysis.get("para1", "").strip()
        para2 = analysis.get("para2", "").strip()
        para3 = analysis.get("para3", "").strip()
        para4 = analysis.get("para4", "").strip()

        # explanation_zh 組成一個大字串（保留給現在或未來前端使用）
        explanation_parts = [p for p in [para1, para2, para3, para4] if p]
        explanation_zh = "\n\n".join(explanation_parts)

        processed.append({
            "id": item["id"],
            "title_en": item["title_en"],
            "pub_date": item["pub_date"],
            "abstract_en": abstract_en,
            "para1": para1,
            "para2": para2,
            "para3": para3,
            "para4": para4,
            "explanation_zh": explanation_zh,
            "is_human_study": analysis.get("is_human_study", False),
            "is_animal_study": analysis.get("is_animal_study", False),
            "is_disease_related": analysis.get("is_disease_related", False),
            "disease_name": analysis.get("disease_name", ""),
            "is_clinical_trial": analysis.get("is_clinical_trial", False),
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

