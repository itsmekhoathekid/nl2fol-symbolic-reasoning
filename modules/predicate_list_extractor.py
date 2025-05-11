from .base import chatAgent
import numpy as np
from sentence_transformers import util

import re

# Danh sách các từ cần loại bỏ nếu đứng đầu
AUX_MODALS = [
    "am", "is", "are", "was", "were", "be", "do", "does", "did",
    "must", "should", "shall", "will", "would", "can", "could", "may", "might", "ought to"
]

NEGATIONS = [
    "am not", "is not", "isn't", "are not", "aren't", "was not", "wasn't", "were not", "weren't",
    "do not", "don't", "does not", "doesn't", "did not", "didn't",
    "must not", "mustn't", "should not", "shouldn't", "shall not", "shan't",
    "will not", "won't", "would not", "wouldn't", "cannot", "can't",
    "could not", "couldn't", "may not", "might not", "ought not to","have" ,"has", "had" ,"having","not"
]

# Gộp hai danh sách lại để xử lý chung
REMOVE_PATTERNS = NEGATIONS + AUX_MODALS

# Tạo regex để nhận diện nếu chuỗi bắt đầu bằng các mẫu trên
remove_regex = re.compile(rf"^({'|'.join(re.escape(p) for p in sorted(REMOVE_PATTERNS, key=len, reverse=True))})\s+", re.IGNORECASE)

def clean_predicate(phrase):
    return remove_regex.sub('', phrase.strip())

# Tách nếu có "and"/"or"
def split_compound(phrase):
    return [p.strip() for p in re.split(r'\band\b|\bor\b', phrase)]

# Tổng xử lý
def extract_clean_predicates(raw_phrase):
    predicates = []
    for part in split_compound(raw_phrase):
        cleaned = clean_predicate(part)
        if cleaned:
            predicates.append(cleaned)
    return predicates[0]

def check_multiple_choice(question: str):
    if len(question.split('\n1 ')) > 2:
        return True
    return False

class predicate_nl_extractor(chatAgent):
    def __init__(self, pipeline_mistral, mapping_model, threshold=0.6):
        super().__init__(pipeline_mistral)
        self.model = pipeline_mistral
        self.mapping_model = mapping_model
        self.threshold = threshold
    
    def extract(self, premise_list, conclusion_list):
        premise_pred_dic = {}
        subject_pred_dic = {}

        pred_list = []
        sub_list = []
        nl_total = premise_list + conclusion_list
        nl_total_list = []

        for idx, premise_nl in enumerate(nl_total):
            split_items = []
            if idx >= len(premise_list)-1 and check_multiple_choice(premise_nl):
                split_items = premise_nl.split('\n1 ')
            else:
                split_items = [premise_nl]

            for item in split_items:
                # Extract predicate
                predicate_prompt = self.get_predicate_from_nl(item)

                response = self.get_response(predicate_prompt)
                extracted_preds = self.extract_predicates_from_table(response)
                # print(response)
                extracted_preds = [extract_clean_predicates(pred) for pred in extracted_preds]

                

                premise_pred_dic[item] = extracted_preds
                pred_list.extend(extracted_preds)

                # Identify subject
                subject_prompt = self.identify_subject_prompt(item)
                response_subject = self.model(subject_prompt, return_full_text=False, max_new_tokens=10)[0]['generated_text'].strip().splitlines()[0]
                subject_pred_dic[item] = response_subject
                sub_list.append(response_subject)

                nl_total_list.append(item)

                print(item)
                print(extracted_preds)

        # Clustering predicates
        pred_mapping = {}
        pred_list_final = []
        mapping_matrix = self.clustering(pred_list)
        for group in mapping_matrix:
            for pred in group:
                pred_mapping[pred] = group[0]
            if len(group) > 0:
                pred_list_final.append(group[0])

        # Clustering subjects
        sub_mapping = {}
        sub_list_final = []
        mapping_matrix_sub = self.clustering(sub_list)
        for group in mapping_matrix_sub:
            for sub in group:
                sub_mapping[sub] = group[0]
            if len(group) > 0:
                sub_list_final.append(group[0])

        # Map predicates to canonical form
        for nl in nl_total_list:
            premise_pred_dic[nl] = [pred_mapping.get(pred, pred) for pred in premise_pred_dic[nl]]

        # Map subject to canonical form
        for nl in nl_total_list:
            raw_sub = subject_pred_dic[nl]
            subject_pred_dic[nl] = sub_mapping.get(raw_sub, raw_sub)

        return premise_pred_dic, pred_list_final, subject_pred_dic, sub_list_final
        # map premise nl over predicates

    def get_predicate_from_nl(self, sentence):
        prompt = f"""
    You are a symbolic reasoning assistant.

    ### Task:
    Given a natural language statement (premise), extract all *positive predicate phrases* that describe main actions or properties. Focus only on actions or qualities expressed — remove anything that is negative or auxiliary.

    ---

    ### Instructions:

    1. **Extract only predicate phrases** — no subject (e.g., "students", "Python project", etc.).
    2. **Only keep positive statements**:
    - ⚠️ Remove all negations like: "not", "does not", "do not", "is not", "was not", "aren’t", "won’t", etc.
    - ✂ Example: "does not follow PEP 8 standards" → "follow PEP 8 standards"
    - ✂ Example: "is not optimized" → "optimized"
    3. **Remove auxiliaries/modals**: am, is, are, was, were, be, do, does, did, must, can, should, etc.
    4. **Split compound predicates**: Separate items joined by "and"/"or"
    5. **Preserve full meaning**: Keep key complements like "by the team", "for graduation", etc.

    ---

    ### Output Format (strict, no explanation, separate each predicate with a | :
    | [Predicate 1]
    | [Predicate 2]
    ...

    ### Premise:
    {sentence}
    ### Output:
    """
        return prompt



    def identify_subject_prompt(self, nl_premise):
        return f"""
        ### Task: Identify the correct logical subject in the given natural language premise.

        You are given a premise sentence that may describe:
        - A general rule (e.g., "If a student submits an assignment...")
        - A specific fact about an individual (e.g., "John submitted the assignment.")

        Your job is to decide whether the subject of the premise should be:
        - `x` → if the sentence describes a general case about a generic person.
        - the actual **name mentioned** (e.g., "John", "Sophia") → if the sentence refers to a specific known individual.

        ### Rules:
        1. If the subject is a general noun phrase like "a student", "students", "anyone", or "employees", return: `x`
        2. If the subject is a named individual (e.g., "John", "Sophia", "Alice"), return their exact name as it appears.
        3. Return only the subject — no extra text, no explanation.

        ### Input:
        Premise: {nl_premise}

        ### Output:
        """

    
    def extract_predicates_from_table(self, table_str):
        lines = table_str.strip().split("\n")
        # Loại bỏ dấu | ở đầu và cuối, rồi tách lấy các predicate
        predicate_lines = [line.replace("|", "").strip() for line in lines if "|" in line and not "Predicate" in line and not "----" in line]
        # Đảm bảo rằng predicate không có dấu | và khoảng trắng thừa
        predicates = [p.strip().lower() for p in predicate_lines if p]
        return predicates

    
    

    def clustering(self, pred_nl_list):
        lps_list = list(pred_nl_list)  # convert set to list
        definitions = [lp.strip() for lp in lps_list]
        embeddings = self.mapping_model.encode(definitions, convert_to_tensor=True)
        list_cosine_scores = util.cos_sim(embeddings, embeddings)
        list_cosine_scores = [scores.detach().cpu() for scores in list_cosine_scores]
        list_idxs = [np.where(cosine_scores > self.threshold)[0] for cosine_scores in list_cosine_scores]
        select_lps = [list(np.array(lps_list)[idxs]) for idxs in list_idxs]
        unique_lps = list(map(list, set(tuple(x) for x in select_lps)))
        # unique_lps = [pred_group for pred_group in unique_lps if len(pred_group) > 1]
        return unique_lps

