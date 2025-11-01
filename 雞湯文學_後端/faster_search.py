import torch
from sentence_transformers import util
import pandas as pd

# 讀 CSV（只有 text、label）
df = pd.read_csv("data/chicken_soup.csv")

# 讀取之前存好的向量
vectors = torch.load("data/chicken_soup_vectors.pt")

# 把向量重新放入 DataFrame
df['vector'] = vectors

# 使用者 mood
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
user_mood = "我生活費沒了,花完了,要吃土了"
mood_vector = model.encode(user_mood, convert_to_tensor=True)

# 計算相似度
df['similarity'] = df['vector'].apply(lambda x: util.cos_sim(x, mood_vector).item())

# 顯示最相似的雞湯
top_chicken_soups = df.sort_values(by='similarity', ascending=False).head(5)
print(top_chicken_soups[['text', 'label', 'similarity']])
