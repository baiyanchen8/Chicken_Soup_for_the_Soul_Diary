import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import subprocess
import json
import time, re
import os

# === Ollama CLI 標註函數 ===
def run_ollama_command(command, user_input):
    """透過 subprocess 執行 ollama run 並回傳 stdout"""
    full_command = f'echo "{user_input}" | {command}'
    start_time = time.time()
    process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()
    response_time = end_time - start_time
    return stdout.decode('utf-8').strip(), response_time

def download_model(model_name):
    """檢查並下載模型"""
    print(f"Checking if the model '{model_name}' is already downloaded...")
    download_command = f"ollama pull {model_name}"
    process = subprocess.Popen(download_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print(f"Model '{model_name}' has been successfully downloaded.")
    else:
        print(f"Failed to download the model '{model_name}'. Error: {stderr.decode('utf-8')}")
        exit(1)

def extract_json_from_text(text, default_value=3):
    """
    從文字中抓取第一個 JSON 物件，解析為字典
    text: 模型回傳的文字
    default_value: 若抓不到或解析失敗，填入預設值
    """
    try:
        # 抓出第一個 { ... } 的部分
        match = re.findall(r'"(\w+_level)":\s*([1-5])', text)
        json_match = {k:int(v) for k,v in match}
        print(f"Extracted JSON: {json_match}")
        if len(json_match) == 4:
            return json_match
    except Exception as e:
        print(f"⚠️ JSON解析失敗: {e} | 原始文字: {text[:100]}...")
    
    # 失敗就回傳 None，讓呼叫端知道需要重試
    return None

def run_ollama_with_retry(command, prompt, max_retries=3, retry_delay=1):
    """
    帶有重試機制的 Ollama 執行函數
    command: ollama 命令
    prompt: 輸入提示
    max_retries: 最大重試次數
    retry_delay: 重試間隔（秒）
    """
    for attempt in range(max_retries):
        response, rt = run_ollama_command(command, prompt)
        labels = extract_json_from_text(response)
        
        if labels is not None:
            return labels, rt, response
        else:
            print(f"⚠️ 第 {attempt + 1} 次嘗試失敗，{f'{retry_delay}秒後重試...' if attempt < max_retries - 1 else '已達最大重試次數'}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    # 所有重試都失敗，回傳預設值
    print("❌ 所有重試失敗，使用預設值")
    default_labels = {
        'stress_level': 3,
        'happiness_level': 3,
        'humor_level': 3,
        'encouragement_level': 3
    }
    return default_labels, 0, ""

def save_progress(df, vectors, checkpoint_dir="./data", prefix="chicken_soup"):
    """
    定期儲存進度到 ./data 目錄
    """
    # 建立檢查點目錄
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 儲存 CSV 檔案
    csv_filename = f"{prefix}_checkpoint_{timestamp}.csv"
    csv_path = os.path.join(checkpoint_dir, csv_filename)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 儲存向量檔案
    vector_filename = f"{prefix}_vectors_checkpoint_{timestamp}.pt"
    vector_path = os.path.join(checkpoint_dir, vector_filename)
    torch.save(vectors, vector_path)
    
    print(f"💾 檢查點已儲存到 {checkpoint_dir}: {csv_filename}, {vector_filename}")

def load_latest_checkpoint(checkpoint_dir="./data", prefix="chicken_soup"):
    """
    從 ./data 目錄載入最新的檢查點
    """
    if not os.path.exists(checkpoint_dir):
        return None, None, 0
    
    # 尋找最新的 CSV 檢查點檔案
    csv_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(".csv") and "checkpoint" in f]
    if not csv_files:
        return None, None, 0
    
    # 按時間排序，取得最新的檔案
    csv_files.sort(reverse=True)
    latest_csv = os.path.join(checkpoint_dir, csv_files[0])
    
    # 對應的向量檔案
    vector_file = csv_files[0].replace(".csv", ".pt").replace("_checkpoint_", "_vectors_checkpoint_")
    latest_vector = os.path.join(checkpoint_dir, vector_file)
    
    if os.path.exists(latest_vector):
        df = pd.read_csv(latest_csv)
        vectors = torch.load(latest_vector)
        
        # 計算已處理的數量（排除 None 值）
        processed_count = df[df['stress_level'].notna()].shape[0]
        
        print(f"🔄 從檢查點恢復: {latest_csv} (已處理 {processed_count} 條)")
        return df, vectors, processed_count
    
    return None, None, 0

# === 主程式 ===
if __name__ == "__main__":
    model_name = "gemma3:4b"
    save_interval = 100  # 每 100 條儲存一次
    data_dir = "./data"  # 所有檔案都放在 ./data 目錄

    # 下載模型
    download_model(model_name)
    command = f"ollama run {model_name}"

    # === 讀入雞湯資料 ===
    csv_path = os.path.join(data_dir, "chicken_soup.csv")
    df = pd.read_csv(csv_path)
    for col in ['stress_level', 'happiness_level', 'humor_level', 'encouragement_level', 'vector']:
        if col not in df.columns:
            df[col] = None

    # === 初始化 SentenceTransformer 模型 ===
    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    # === 嘗試載入最新的檢查點 ===
    checkpoint_df, checkpoint_vectors, start_idx = load_latest_checkpoint(data_dir)
    if checkpoint_df is not None:
        df = checkpoint_df
        # 更新向量資料
        for i, vector in enumerate(checkpoint_vectors):
            if i < len(df):
                df.at[i, 'vector'] = vector
        print(f"🔄 從第 {start_idx} 條開始繼續處理")
    else:
        start_idx = 0
        print("🚀 開始新的處理任務")

    # === 對每條雞湯標註心理指標 & 計算向量 ===
    total_items = len(df)
    
    for idx in range(start_idx, total_items):
        row = df.iloc[idx]
        text = row['text']

        # 跳過已處理的項目
        if pd.notna(df.at[idx, 'stress_level']):
            continue

        prompt = f"請將下面雞湯標註心理指標（1~5）：- stress_level: 壓力感（1~5） - happiness_level: 開心程度（1~5） - humor_level: 幽默程度（1~5） - encouragement_level: 鼓勵需求（1~5） - 雞湯內容：{text} - 請只回傳 JSON 格式，例如：{{stress_level:?, happiness_level:?, humor_level:?, encouragement_level:?}}"

        # 使用帶有重試機制的函數
        labels, rt, response = run_ollama_with_retry(command, prompt)
        
        df.at[idx, 'stress_level'] = labels['stress_level']
        df.at[idx, 'happiness_level'] = labels['happiness_level']
        df.at[idx, 'humor_level'] = labels['humor_level']
        df.at[idx, 'encouragement_level'] = labels['encouragement_level']

        # 計算文本向量
        df.at[idx, 'vector'] = st_model.encode(text, convert_to_tensor=True)

        # 進度顯示
        if (idx + 1) % 20 == 0:
            print(f"✅ 已處理 {idx + 1}/{total_items} 條雞湯，耗時 {rt:.2f}s")

        # 定期儲存
        if (idx + 1) % save_interval == 0:
            print(f"💾 達到 {idx + 1} 條，進行定期儲存...")
            vectors = df['vector'].tolist()
            save_progress(df, vectors, data_dir)
            print(f"✅ 第 {idx + 1} 條已儲存完成")

    # === 最終儲存資料庫 ===
    print("💾 正在進行最終儲存...")
    
    # 最終向量檔案路徑
    final_vector_path = os.path.join(data_dir, "chicken_soup_vectors.pt")
    torch.save(df['vector'].tolist(), final_vector_path)
    
    # 最終 CSV 檔案路徑
    final_csv_path = os.path.join(data_dir, "chicken_soup_with_features.csv")
    df.to_csv(final_csv_path, index=False, encoding="utf-8-sig")
    
    # 清理檢查點檔案（可選）
    print("🧹 清理檢查點檔案...")
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.startswith("chicken_soup") and "checkpoint" in file:
                file_path = os.path.join(data_dir, file)
                os.remove(file_path)
                print(f"🗑️ 已刪除檢查點檔案: {file}")
    
    print("✅ 雞湯資料庫建立完成，包含心理指標與向量")
    print(f"📁 最終檔案位置:")
    print(f"   - 向量檔案: {final_vector_path}")
    print(f"   - 特徵檔案: {final_csv_path}")