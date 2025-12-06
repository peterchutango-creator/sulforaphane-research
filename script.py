import os
import json
from typing import List, Dict, Any

from Bio import Entrez
from openai import OpenAI

# --- Configuration --------------------------------------------------------

EMAIL = "peter.chu.tango@gmail.com"   # NCBI 要求的 email
SEARCH_TERM = '"sulforaphane"[Title]' # 只搜尋標題含 sulforaphane
MAX_RESULTS = 30                      # 最終要留下的論文數量
OUTPUT_FILE = "data/research.json"

# --- Services -------------------------------------------------------------

Entrez.email = EMAIL
# 如果你有 NCBI API key，可以解開這行
# Entrez.api_key = os.getenv("NCBI_API_KEY")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[Warning] OPENAI_API_KEY 環境變數不存在，將無法產生中文解說與 FB 貼文。")
client = OpenAI(api_key=api_key) if api_key else None


# --- Helpers --------------------------------------------------------------

def clean_json_str(content: str) -> str:
    """
    有時模型會回傳 ```json ... ``` 的格式，
    這裡先把外層 code block 剝掉再丟給 json.loads。
    """
    text = content.strip()
    if text.startswith("```"):
        # 去掉開頭/結尾的 ```
        text = text.strip("`").strip()
        # 可能會是 "json\n{...}"
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def to_zh_year_month(pub_date: str) -> str:
    """
    將類似 '2025 Sep' / '2025 January' / '2025' 轉成
    '2025 年 9 月' / '2025 年 1 月' / '2025 年'。
    如果看不懂，就原樣回傳。
    """
    if not pub_date:
        return ""

    tokens = pub_date.split()
    year = ""
    month = ""

    if len(tokens) >= 2 and tokens[0].isdigit():
        year = tokens[0]
        month_raw = tokens[1]
    elif len(tokens) == 1 and tokens[0].isdigit():
        year = tokens[0]
        month_raw = ""
    else:
        # 無法解析，直接原樣回傳
        return pub_date

    month_map = {
        "Jan": "1", "January": "1",
        "Feb": "2", "February": "2",
        "Mar": "3", "March": "3",
        "Apr": "4", "April": "4",
        "May": "5",
        "Jun": "6", "June": "6",
        "Jul": "7", "July": "7",
        "Aug": "8", "August": "8",
        "Sep": "9", "Sept": "9", "September": "9",
        "Oct": "10", "October": "10",
        "Nov": "11", "November": "11",
        "Dec": "12", "December": "12",
    }

    month_num = ""
    if len(tokens) >= 2:
        month_num = month_map.get(tokens[1], "")

    if year and month_num:
        return f"{year} 年 {month_num} 月"
    if year:
        return f"{year} 年"
    return pub_date


# --- OpenAI：解析摘要成四段中文 + 類別標記 -----------------------------

def analyze_abstract(abstract_en: str) -> Dict[str, Any]:
    """
    使用 OpenAI：
    1. 產出大人小孩看得懂的中文解說，拆成 para1–para4 四段
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


# --- OpenAI：從 para1–4 產 FB 貼文 + 產圖 Prompt -------------------------

def generate_social_content(para1: str, para2: str, para3: str, para4: str) -> Dict[str, Any]:
    """
    根據四段中文科普（para1~para4），產生：
    - fb_post：適合貼在 FB 的貼文（繁體中文）
    - image_prompt：一段英文產圖 Prompt，重點在「實驗方式 + 結果」
    """
    merged = "\n\n".join([p for p in [para1, para2, para3, para4] if p])

    # 沒有 client 時直接 fallback
    if client is None:
        return {
            "fb_post": f"（自動產生 FB 貼文失敗，以下為原始科普內容）\n\n{merged}",
            "image_prompt": (
                "Design a clean, flat-style scientific illustration that explains a sulforaphane study. "
                "Use simple icons to show the experiment subjects (humans / animals / cells), "
                "how sulforaphane was given, and what main result was observed. "
                "Use soft colors, white or light background, clear arrows to show the flow of the experiment, "
                "and a small area highlighting the key outcome (for example, better protection, reduced inflammation, "
                "or improved cellular response). Avoid complex chemical structures or realistic medical photos."
                "翻成繁體中文，字要夠大方便閱讀“
            )
        }

    prompt = f"""
你是一位擅長寫社群貼文和協助設計師做圖的健康科普創作者。

以下是同一篇蘿蔔硫素論文的四段中文科普內容：
para1：{para1}
para2：{para2}
para3：{para3}
para4：{para4}

請你幫我完成兩件事：

1. 把這篇論文改寫成一篇適合貼在 Facebook 粉絲專頁的貼文內容（fb_post）：
   - 用「像在跟朋友聊天」的口吻，讓一般大人和小孩都看得懂、願意看完
   - 開頭用一個有記憶點的 hook（不要太廣告感）
   - 中間簡單交代：研究在關心什麼、怎麼做、觀察到什麼結果
   - 避免專有名詞，如果有專有名詞，一定要簡短解釋
   - 結尾給讀者 2～3 個重點（條列式）
   - 字數控制在 300～450 字
   - 不要說「可以治癒」「一定有效」，保持謹慎、科學的語氣

2. 為這篇貼文寫一段「產圖 Prompt」（image_prompt），給 AI 畫圖工具或設計師使用：
   - 圖片的目標：用簡單易懂的方式解釋「這篇論文的實驗方式和主要結果」
   - 風格：扁平化圖示、白底或淺色背景、柔和配色、科普／資訊圖風格
   - 畫面可以包含：
       - 青花椰菜或蘿蔔硫素的象徵圖示
       - 實驗對象（例如人、動物、細胞）的簡化圖示
       - 箭頭或流程線，表示實驗步驟
       - 一區塊簡單標示「主要結果」（例如某種保護效果、變化方向等）
   - 請用一段完整的英文描述（因為多數繪圖工具較適合英文 Prompt）
   - 不要提到這是為了 Facebook 或社群貼文，只專注在圖像內容

⚠ 請你最後只輸出一個 JSON 物件，不要有多餘說明文字。
JSON 格式必須長這樣：

{{
  "fb_post": "這裡是一篇可直接貼在 Facebook 的貼文內容（繁體中文）。",
  "image_prompt": "這裡是一段用英文寫的圖片生成 prompt。"
}}

記得：
- 只輸出一個 JSON 物件
- 不要多加註解、說明、程式碼區塊
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.5,
        )
        content = resp.choices[0].message.content or ""
        content = clean_json_str(content)
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"[OpenAI / FB 貼文生成錯誤] {e}")
        return {
            "fb_post": f"（自動產生 FB 貼文失敗，以下為原始科普內容）\n\n{merged}",
            "image_prompt": (
                "Design a clean, flat-style scientific illustration that explains a sulforaphane study. "
                "Use simple icons to show the experiment subjects (humans / animals / cells), "
                "how sulforaphane was given, and what main result was observed. "
                "Use soft colors, white or light background, clear arrows to show the flow of the experiment, "
                "and a small area highlighting the key outcome (for example, better protection, reduced inflammation, "
                "or improved cellular response). Avoid complex chemical structures or realistic medical photos."
            )
        }


# --- PubMed 抓資料 -------------------------------------------------------

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

            # --- 期刊名稱（論文出處） ---
            journal_title = str(article_data.get("Journal", {}).get("Title", "")).strip()

            papers.append({
                "id": pmid,
                "title_en": title_en,
                "abstract_en": abstract_en,
                "pub_date": pub_date,
                "journal": journal_title,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        except Exception as e:
            print(f"Skipping one article due to parsing error: {e}")
            continue

    # 最後只留下最新的 MAX_RESULTS 篇
    papers = papers[:MAX_RESULTS]

    print(f"Fetched {len(papers)} valid articles from PubMed.")
    return papers


# --- 組合 + 寫入 JSON ----------------------------------------------------

def process_and_save_data(raw_data: List[Dict[str, Any]]) -> None:
    """
    對每篇論文呼叫 OpenAI：
    - 產生四段中文白話解說（para1~para4）
    - 組合 explanation_zh（給原本頁面用）
    - 產生 FB 貼文（fb_post），並在開頭加上「根據XXX於XX年XX月發表的論文，」
    - 產生圖片 prompt（image_prompt）
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

        explanation_parts = [p for p in [para1, para2, para3, para4] if p]
        explanation_zh = "\n\n".join(explanation_parts)

        # 根據 para1~para4 產 FB 貼文 + 圖片 prompt（原始內容）
        social = generate_social_content(para1, para2, para3, para4)
        fb_post_raw = social.get("fb_post", "") or ""
        image_prompt = social.get("image_prompt", "") or ""

        # 論文出處 + 日期，組成 FB 前綴
        journal = item.get("journal", "").strip()
        pub_date = item.get("pub_date", "").strip()
        date_phrase = to_zh_year_month(pub_date)

        if journal and date_phrase:
            prefix = f"根據《{journal}》於 {date_phrase} 發表的論文，"
        elif date_phrase:
            prefix = f"根據於 {date_phrase} 發表的論文，"
        elif journal:
            prefix = f"根據發表於《{journal}》的論文，"
        else:
            prefix = "根據這篇蘿蔔硫素研究論文，"

        fb_post = prefix + fb_post_raw.lstrip()

        processed.append({
            "id": item["id"],
            "title_en": item["title_en"],
            "pub_date": pub_date,
            "journal": journal,
            "abstract_en": abstract_en,
            "para1": para1,
            "para2": para2,
            "para3": para3,
            "para4": para4,
            "explanation_zh": explanation_zh,
            "fb_post": fb_post,
            "image_prompt": image_prompt,
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


# --- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    raw = fetch_pubmed_data()
    if raw:
        process_and_save_data(raw)

