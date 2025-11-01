import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# === 初始化模型 ===
print("🚀 載入模型中，請稍候...")
model = SentenceTransformer('all-MiniLM-L6-v2')

try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased")
except Exception as e:
    print("⚠️ 中文情緒分析模型載入失敗，使用英文預設模型。")
    sentiment_analyzer = pipeline("sentiment-analysis")

# === 載入雞湯資料與向量 ===
df = pd.read_csv("data/chicken_soup_with_features.csv")

# 修正：使用 map_location 載入向量檔案
try:
    vectors = torch.load("data/chicken_soup_vectors.pt", map_location=torch.device('cpu'))
    print("✅ 向量檔案載入成功")
except Exception as e:
    print(f"❌ 向量檔案載入失敗: {e}")
    print("🔧 重新計算向量...")
    # 如果向量檔案載入失敗，重新計算向量
    vectors = []
    for idx, text in enumerate(df['text']):
        vector = model.encode(text, convert_to_tensor=True)
        vectors.append(vector.cpu())  # 確保在 CPU 上
        if (idx + 1) % 100 == 0:
            print(f"✅ 已計算 {idx + 1} 個向量")
    
    # 儲存新的向量檔案
    torch.save(vectors, "data/chicken_soup_vectors.pt")
    print("✅ 新的向量檔案已儲存")

# 確保所有向量都在 CPU 上並轉換為列表
if isinstance(vectors, torch.Tensor):
    vectors = [vec.cpu() for vec in vectors]
else:
    vectors = [vec.cpu() if isinstance(vec, torch.Tensor) else vec for vec in vectors]

df['vector'] = vectors

print("✅ 資料與模型已載入完成！\n")

# === 功能選單 ===
print("==== 🐣 智慧雞湯推薦系統 ====")
print("1️⃣ 問卷模式：我自己選要聽雞湯或毒雞湯")
print("2️⃣ 自動模式：AI 幫我判斷心情，自動推薦雞湯")
print("3️⃣ 屬性模式：根據心理屬性推薦雞湯\n")

mode = input("請選擇模式（輸入 1、2 或 3）：").strip()

# === 使用者心情輸入 ===
user_mood = input("\n請描述你現在的心情：")

# === 模式 1：問卷模式 ===
if mode == "1":
    user_prefer = input("你想聽【雞湯】還是【毒雞湯】？(輸入 positive 或 negative)：").strip().lower()
    if user_prefer not in ["positive", "negative"]:
        print("⚠️ 輸入錯誤，預設為 positive（正向雞湯）")
        user_prefer = "positive"
    prefer = user_prefer

# === 模式 2：自動判斷情緒 ===
elif mode == "2":
    try:
        sentiment = sentiment_analyzer(user_mood)[0]
        label = sentiment['label']
        score = sentiment['score']

        print(f"\n🧠 模型判斷你的情緒為：{label}（信心值 {score:.2f}）")

        # 若為負面 → 推正向雞湯；若為正面 → 推毒雞湯
        if "NEG" in label.upper():
            prefer = "positive"
        elif "POS" in label.upper():
            prefer = "negative"
        else:
            prefer = "positive"

        print(f"📘 系統決定為你推薦：{prefer} 雞湯\n")
    except Exception as e:
        print(f"⚠️ 情緒分析失敗: {e}，預設使用正向雞湯")
        prefer = "positive"

# === 模式 3：屬性模式 ===
elif mode == "3":
    print("\n🎯 屬性模式：請為以下心理屬性評分（1-5分）：")
    
    try:
        stress_input = input("你目前的壓力程度 (1-5，1=很低，5=很高)：").strip()
        happiness_input = input("你希望的開心程度 (1-5，1=很低，5=很高)：").strip()
        humor_input = input("你希望的幽默程度 (1-5，1=很低，5=很高)：").strip()
        encouragement_input = input("你需要的鼓勵程度 (1-5，1=很低，5=很高)：").strip()
        
        # 轉換為整數，如果輸入無效則使用預設值
        user_stress = int(stress_input) if stress_input.isdigit() and 1 <= int(stress_input) <= 5 else 3
        user_happiness = int(happiness_input) if happiness_input.isdigit() and 1 <= int(happiness_input) <= 5 else 3
        user_humor = int(humor_input) if humor_input.isdigit() and 1 <= int(humor_input) <= 5 else 3
        user_encouragement = int(encouragement_input) if encouragement_input.isdigit() and 1 <= int(encouragement_input) <= 5 else 3
        
        print(f"\n📊 你的屬性設定：")
        print(f"  壓力程度: {user_stress}")
        print(f"  開心程度: {user_happiness}")
        print(f"  幽默程度: {user_humor}")
        print(f"  鼓勵程度: {user_encouragement}")
        
    except Exception as e:
        print(f"⚠️ 輸入格式錯誤，使用預設屬性值。錯誤：{e}")
        user_stress, user_happiness, user_humor, user_encouragement = 3, 3, 3, 3

else:
    print("⚠️ 未選擇有效模式，預設為問卷模式（正向雞湯）")
    prefer = "positive"

# === 根據不同模式進行推薦 ===
if mode == "3":
    # === 屬性模式：計算屬性匹配度 ===
    print("\n🔍 正在根據屬性匹配度推薦雞湯...")
    
    # 計算屬性差異的絕對值總和（差異越小越匹配）
    df['attribute_match'] = df.apply(
        lambda row: (
            abs(row['stress_level'] - user_stress) +
            abs(row['happiness_level'] - user_happiness) +
            abs(row['humor_level'] - user_humor) +
            abs(row['encouragement_level'] - user_encouragement)
        ), axis=1
    )
    
    # 同時也計算文本相似度
    mood_vector = model.encode(user_mood, convert_to_tensor=True).cpu()
    df['similarity'] = df['vector'].apply(
        lambda x: util.cos_sim(x.cpu() if isinstance(x, torch.Tensor) else x, mood_vector).item()
    )
    
    # 綜合評分：屬性匹配度（權重0.7） + 文本相似度（權重0.3）
    df['combined_score'] = (1 - df['attribute_match'] / 16) * 0.7 + df['similarity'] * 0.3
    
    # 取前5名
    top_chicken_soups = df.sort_values(by='combined_score', ascending=False).head(5)
    
    # === 輸出結果 ===
    print("===== 🎯 根據屬性為你推薦的雞湯 =====")
    for i, row in top_chicken_soups.iterrows():
        print(f"\n[綜合評分: {row['combined_score']:.3f}]")
        print(f"屬性匹配: {1 - row['attribute_match']/16:.3f}, 文本相似: {row['similarity']:.3f}")
        print(f"壓力:{row['stress_level']} 開心:{row['happiness_level']} 幽默:{row['humor_level']} 鼓勵:{row['encouragement_level']}")
        print(f"👉 {row['text']}")

else:
    # === 模式1和2：傳統推薦方式 ===
    mood_vector = model.encode(user_mood, convert_to_tensor=True).cpu()
    
    # 過濾雞湯類別
    filtered_df = df[df['label'] == prefer].copy()
    
    # 計算相似度
    filtered_df['similarity'] = filtered_df['vector'].apply(
        lambda x: util.cos_sim(x.cpu() if isinstance(x, torch.Tensor) else x, mood_vector).item()
    )
    
    # 取前5名
    top_chicken_soups = filtered_df.sort_values(by='similarity', ascending=False).head(5)
    
    # === 輸出結果 ===
    mode_name = "問卷模式" if mode == "1" else "自動模式"
    print(f"===== 🍵 {mode_name}為你推薦的雞湯 =====")
    for i, row in top_chicken_soups.iterrows():
        print(f"\n[{row['label']}] 相似度: {row['similarity']:.3f}")
        print(f"壓力:{row['stress_level']} 開心:{row['happiness_level']} 幽默:{row['humor_level']} 鼓勵:{row['encouragement_level']}")
        print(f"👉 {row['text']}")

print("\n✨ 推薦完成！祝你心情更好 💖")