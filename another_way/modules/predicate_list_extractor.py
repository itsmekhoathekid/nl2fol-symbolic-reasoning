from .base import chatAgent
import numpy as np
from sentence_transformers import util
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk.tree import ParentedTree
import re
import nltk

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


def extract_subject (parse_tree):
    # Extract the first noun found in NP_subtree
    subject = []
    for s in parse_tree.subtrees(lambda x: x.label() == 'NP'):
        for t in s.subtrees(lambda y: y.label().startswith('NN')):
            output = [t[0], extract_attr(t)]
            # Avoid empty or repeated values
            if output != [] and output not in subject:
                subject.append(output)
    if len(subject) != 0: return subject[0]
    else: return ['']

def extract_lowest_level_predicate(parse_tree):
    predicates = []

    for vp in parse_tree.subtrees(lambda t: t.label() == 'VP'):
        # Nếu subtree này KHÔNG chứa bất kỳ VP con nào khác (ngoài chính nó)
        has_descendant_vp = any(
            sub.label() == 'VP' and sub != vp
            for sub in vp.subtrees()
        )

        if not has_descendant_vp:
            predicate_phrase = ' '.join(vp.leaves())
            predicates.append(predicate_phrase)

    return predicates



def extract_object (parse_tree):
    # Extract the first noun or first adjective in NP, PP, ADP siblings of VP_subtree
    objects, output, word = [],[],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label() in ['NP','PP','ADP']):
            if t.label() in ['NP','PP']:
                for u in t.subtrees(lambda z: z.label().startswith('NN')):
                    word = u
            else:
                for u in t.subtrees(lambda z: z.label().startswith('JJ')):
                    word = u
            if len(word) != 0:
                output = [word[0], extract_attr(word)]
            if output != [] and output not in objects:
                objects.append(output)
    if len(objects) != 0: return objects[0]
    else: return ['']

def extract_attr (word):
    attrs = []
    # Search among the word's siblings
    if word.label().startswith('JJ'):
        for p in word.parent():
            if p.label() == 'RB':
                attrs.append(p[0])
    elif word.label().startswith('NN'):
        for p in word.parent():
            if p.label() in ['DT','PRP$','POS','JJ','CD','ADJP','QP','NP']:
                attrs.append(p[0])
    elif word.label().startswith('VB'):
        for p in word.parent():
            if p.label() == 'ADVP':
                attrs.append(p[0])
    # Search among the word's uncles
    if word.label().startswith('NN') or word.label().startswith('JJ'):
        for p in word.parent().parent():
            if p.label() == 'PP' and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    elif word.label().startswith('VB'):
        for p in word.parent().parent():
            if p.label().startswith('VB') and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    return attrs


def remove_substrings(lst):
    result = []
    for item in lst:
        if not any((item != other and item in other) for other in lst):
            result.append(item)
    return result

redundant_list = ['then' ,'that']
def remove_redundant(input):
  for word in redundant_list:
    input = input.replace(word, '')
  return input.strip()

class predicate_nl_extractor(chatAgent):
    def __init__(self, pipeline_mistral, mapping_model, threshold=0.6, port = 8000):
        super().__init__(pipeline_mistral)
        self.model = pipeline_mistral
        self.mapping_model = mapping_model
        self.threshold = threshold
        self.dep_parser = CoreNLPDependencyParser(url='http://0.0.0.0:8000')
        self.pos_tagger = CoreNLPParser(url='http://0.0.0.0:8000', tagtype='pos')
    
    def triplet_extraction(self, input_sent, output=['parse_tree','spo','result']):
        input_sent = remove_redundant(input_sent)
        # Parse the input sentence with Stanford CoreNLP Parser
        pos_type = self.pos_tagger.tag(input_sent.split())
        parse_tree, = ParentedTree.convert(list(self.pos_tagger.parse(input_sent.split()))[0])
        dep_type, = ParentedTree.convert(self.dep_parser.parse(input_sent.split()))


        # pos_dict = {word: tag for word, tag in pos_type}
        # # input_sent = ' '.join([word for word, tag in pos_dict.items() if tag not in ['RB']])


        # pos_type = pos_tagger.tag(input_sent.split())
        # parse_tree, = ParentedTree.convert(list(pos_tagger.parse(input_sent.split()))[0])
        # dep_type, = ParentedTree.convert(dep_parser.parse(input_sent.split()))

        pos_dict = {word: tag for word, tag in pos_type}
        def extract_words_with_joined_nns(inputs, pos_type):
            inputs = inputs.replace(',', '')
            inputs = inputs.replace('.', '')
            input_split = inputs.split()

            result = []
            buffer = []

            def flush_buffer():
                if buffer:
                    result.append('_'.join(buffer))
                    buffer.clear()

            for word in input_split:
                if word in pos_type and 'NN' in pos_type[word]:
                    buffer.append(word)
                else:
                    flush_buffer()
                    result.append(word)
            flush_buffer()
            return result

        # print(' '.join(extract_words_with_joined_nns(input_sent.split(), pos_dict)))
        input_sent = ' '.join(extract_words_with_joined_nns(input_sent, pos_dict))
        # Parse the input sentence with Stanford CoreNLP Parser

        pos_type = self.pos_tagger.tag(input_sent.split())
        parse_tree, = ParentedTree.convert(list(self.pos_tagger.parse(input_sent.split()))[0])
        dep_type, = ParentedTree.convert(self.dep_parser.parse(input_sent.split()))


        # Extract subject, predicate and object
        subject = extract_subject(parse_tree)
        predicates = extract_lowest_level_predicate(parse_tree)
        objects = extract_object(parse_tree)



        # pos_type = pos_tagger.tag(input_sent.split())
        # # parse tree (ParentedTree)
        # parse_tree, = ParentedTree.convert(list(pos_tagger.parse(input_sent.split()))[0])
        # dep_type, = ParentedTree.convert(dep_parser.parse(input_sent.split()))

        # Tạo dict từ -> loại POS (từ pos_type)
        pos_dict = {word: tag for word, tag in pos_type}


        # In ra theo yêu cầu output
        if 'parse_tree' in output:
            print('---Parse Tree---')
            tree = parse_tree.pretty_print()

        subject_P = {word:tpe for word, tpe in pos_type if tpe == 'NNP'}
        # vbp_list = {word : tpe for word, tpe in pos_type if tpe in ['VBP'] }


        def remove_vbp(pred):
            for word, tpe in subject_P.items():
                pred = pred.replace(word, '')
            return pred.strip()

        def post_process(predicates):
            res = []
            for i in range(len(predicates)):
                res.append(remove_vbp(predicates[i]))
                if len(predicates[i].split(' ')) == 1 and pos_dict[predicates[i]] == 'VBZ':
                    res.pop()
            return res

        predicates = post_process(predicates)



        

        # Trả về kết quả (subject, predicate, object)
        return subject, predicates, pos_dict, subject_P

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
                try:
                    response_subject, extracted_preds, dic, subP = self.triplet_extraction(item)
                except Exception as e:
                    print(f"Error processing item: {item}")
                    print(f"Exception: {e}")
                    continue
                    
                extracted_preds = remove_substrings(extracted_preds)
                

                premise_pred_dic[item] = extracted_preds
                pred_list.extend(extracted_preds)

                
                subject_pred_dic[item] = response_subject[0]
                sub_list.append(response_subject[0])

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
        
        # print(pred_mapping, flush = True)

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
    6. **Do not** invent new predicates or add any extra text.
    ---

    ### Output Format (strict, no explanation, separate each predicate with a | ):
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

