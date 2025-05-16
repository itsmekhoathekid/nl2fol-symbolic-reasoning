import os, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)
import yaml
from sentence_transformers import SentenceTransformer, util

def load_pipeline(model_path):


    MODEL_NAME = "/data/npl/ViInfographicCaps/Contest/demo_contest/xai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,      
        device_map="auto",          
        use_auth_token=True,
    )

    gen_cfg = GenerationConfig(
        task               = "text-generation",
        max_new_tokens     = 1024,
        do_sample          = True,
        temperature        = 0.1,
        top_p              = 0.95,
        repetition_penalty = 1.15,
        num_beams          = 2,        
        use_cache          = True,
    )

    llama = pipeline(
        "text-generation",
        model      = model,
        tokenizer  = tokenizer,
        generation_config = gen_cfg,
    )


    return llama

def load_yml(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return None


import subprocess
import os
import time

def start_corenlp_server(port=9000):
    # Đảm bảo đang ở đúng thư mục chứa .jar
    os.chdir("/data/npl/ViInfographicCaps/Contest/final_contest/another_way/stanford-corenlp-4.5.6")

    # Nếu đã có server chạy thì không chạy lại
    if not os.path.exists("corenlp.pid"):
        # Tạo lệnh java
        cmd = [
            "java", "-mx4g", "-cp", "*",
            "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
            "-port", str(port),
            "-timeout", "15000"
        ]

        # Mở server ở chế độ nền
        with open("corenlp.log", "w") as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            with open("corenlp.pid", "w") as f:
                f.write(str(process.pid))

        print(f"✅ CoreNLP Server started on port {port}")
        time.sleep(5)  # Chờ server khởi động
    else:
        print("⚠️ Server is already running or pid file exists.")

def stop_corenlp_server():
    if os.path.exists("corenlp.pid"):
        with open("corenlp.pid", "r") as f:
            pid = int(f.read())
        os.kill(pid, 9)
        os.remove("corenlp.pid")
        print("🛑 CoreNLP Server stopped.")
    else:
        print("⚠️ No running server found.")