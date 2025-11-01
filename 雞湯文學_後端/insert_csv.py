import os
import pandas as pd

# 資料夾路徑
base_dir = os.path.expanduser("C:\Users\baiya\Documents\GITHUB\aliada//雞湯文學")
categories = {
    "雞湯": "positive",
    "毒雞湯": "negative"
}

all_data = []

for folder, label in categories.items():
    folder_path = os.path.join(base_dir, folder)
    
    # 遍歷 txt 檔
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # 去除空行
                        all_data.append({"text": line, "label": label})

# 轉成 DataFrame
df = pd.DataFrame(all_data)

# 去除重複
df = df.drop_duplicates(subset="text").reset_index(drop=True)

# 建立輸出資料夾
output_dir = os.path.join(base_dir, "data")
os.makedirs(output_dir, exist_ok=True)

# 存成 CSV
output_csv = os.path.join(output_dir, "chicken_soup.csv")
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ 資料清洗完成，共 {len(df)} 筆資料，已存於 {output_csv}")
