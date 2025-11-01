from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')
df=pd.read_csv("data/chicken_soup.csv")
# 計算向量
df['vector'] = df['text'].apply(lambda x: model.encode(x, convert_to_tensor=True))

# 將向量另存，方便後續查詢
torch.save(df['vector'].tolist(), "data/chicken_soup_vectors.pt")
df.to_csv("data/chicken_soup_with_vectors.csv", index=False, encoding="utf-8-sig")
print("✅ 雞湯向量資料庫建立完成")
