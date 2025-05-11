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


