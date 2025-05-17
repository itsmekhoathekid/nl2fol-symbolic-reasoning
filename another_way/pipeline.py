
from utils import load_pipeline, load_yml, extract_predicate_prompt, MAKE_CONCLUSION_FROM_OPTION_QUESTION, start_corenlp_server
from modules import make_conclusion
from modules import predicate_nl_extractor, Extract_Hypothesis, nl_to_fol

import re
import argparse
import yaml
from sentence_transformers import SentenceTransformer, util
import os 
import json
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for ViInfographicCaps")
    parser.add_argument(
        "--config",
        type=str,
        default="mistral-7b-v1.0",
        help="Path to the model directory or model name",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device ID to use for the model (default: 0)",
    )
    return parser.parse_args()

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def check_multiple_choice(question: str):
    if re.findall(r"\n[A-D][\.\)]? (.*?)(?=\n[A-D][\.\)]? |\Z)", question):
        return True
    return False

if __name__ == "__main__":
    # khoi dong server corenlp
    start_corenlp_server(8000)

    args = parse_args()
    config = load_yml(args.config)

    

    pipeline = load_pipeline(config["model_path"])
    model_embedding = SentenceTransformer(config['model_embedding']).cuda(args.device)

    predicate_extractor = predicate_nl_extractor(pipeline, model_embedding, threshold=0.5)

    nl_to_fol_converter = nl_to_fol(pipeline)

    train = load_json(config["data"]["train"])

    extract_hypothesis_another = Extract_Hypothesis(
        pipeline
    )

    for record in train:

        new_questions = []
        for question in record["questions"]:
            if check_multiple_choice(question):
                new_question = make_conclusion(
                    model=pipeline,
                    question=question,
                    config=config
                )
                new_questions.append(new_question + '.')
            else: # Cho các câu hỏi loại khác
                new_question = extract_hypothesis_another.generate_hypothesis(question)
                new_questions.append(new_question + '.')
        record['questions'] = new_questions



        premise_list = record["premises-NL"]

        premise_list = [p.lower() for p in premise_list]
        extract_predicate_prompt_nl = extract_predicate_prompt(premise_list)

        start_time = time.time() 

        premise_pred_dic, preds, subject_pred_dic, subjects = predicate_extractor.extract(premise_list, record['questions'])

        for premise_nl in list(premise_pred_dic.keys()):
            print(f"Premise: {premise_nl}")
            print(f"Predicates: {premise_pred_dic[premise_nl]}")



        fol_pred_dic = nl_to_fol_converter.convert(preds)
        for pred in preds:
            print(f"Predicate: {pred}")
            # fol_pred_dic[pred] = fol_pred_dic[pred] + " ::: " + pred
            print(f"FOL Predicate: {fol_pred_dic[pred]}")


         
        # Convert premise to FOL
        # print(f"FOL Formula: ")
        # fol_formula_dic = {}
        # for premise_nl in list(premise_pred_dic.keys()):
        #     premise_nl_pred = premise_pred_dic[premise_nl]
        #     fol_formula = nl_to_fol_converter.convert_premise_to_fol(premise_nl, premise_nl_pred, fol_pred_dic, subject_pred_dic)
        #     fol_formula_dic[premise_nl] = fol_formula
        #     print(fol_formula)
        
        premise_nl_list = list(premise_pred_dic.keys())
        fol_formula = nl_to_fol_converter.convert_premise_to_fol(premise_nl_list, premise_pred_dic, fol_pred_dic, subject_pred_dic)

        for premise, fol in fol_formula.items():
            print(fol)
            # fol_formula_dic[premise_nl] = fol_formula
            # print(fol_formula)

        # convert question to FOL

        record['fol_pred_dic'] = fol_pred_dic
        record['premise_pred_dic'] = premise_pred_dic
        record['subject_pred_dic'] = subject_pred_dic
        record['fol_formula'] = fol_formula
        # Save the record to a file

        output_file = '/data/npl/ViInfographicCaps/Contest/final_contest/another_way/save/save_data.json'
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        end_time = time.time()    # ⏱️ Kết thúc đo thời gian
        print(f"Execution time: {end_time - start_time:.4f} seconds")




        raise
    
