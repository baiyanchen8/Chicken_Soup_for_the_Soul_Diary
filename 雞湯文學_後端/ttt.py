import time
import subprocess

def run_ollama_command(command, user_input):
    full_command = f'echo "{user_input}" | {command}'
    start_time = time.time()
    process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()
    
    response_time = end_time - start_time
    return stdout.decode('utf-8'), response_time

def download_model(model_name):
    print(f"Checking if the model '{model_name}' is already downloaded...")
    download_command = f"ollama pull {model_name}"
    process = subprocess.Popen(download_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        print(f"Model '{model_name}' has been successfully downloaded.")
    else:
        print(f"Failed to download the model '{model_name}'. Error: {stderr.decode('utf-8')}")
        exit(1)

if __name__ == "__main__":
    model_name = input("請輸入模型名稱: ")
    
    # Step 1: Download the model if not already downloaded
    download_model(model_name)
    
    # Step 2: Prepare the command for running the model
    command = f"ollama run {model_name}"
    
    # Step 3: Begin the Q&A loop
    while True:
        user_input = input(">>> ")
        if user_input.lower() in ["exit", "/bye"]:
            break
        response, response_time = run_ollama_command(command, user_input)
        print(f"Response:\n{response}")
        print(f"Response Time: {response_time:.4f} seconds")