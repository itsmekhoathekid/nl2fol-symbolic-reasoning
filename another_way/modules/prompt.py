import re



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
    "could not", "couldn't", "may not", "might not", "ought not to"
]

# Gộp hai danh sách lại để xử lý chung
REMOVE_PATTERNS = NEGATIONS + AUX_MODALS

# Tạo regex để nhận diện nếu chuỗi bắt đầu bằng các mẫu trên
remove_regex = re.compile(rf"^({'|'.join(re.escape(p) for p in sorted(REMOVE_PATTERNS, key=len, reverse=True))})\s+", re.IGNORECASE)

examples = [
    "should not be ignored by students",
    "must complete the form",
    "did not finish the report",
    "can access the database",
    "are studying at the library",
    "should and must submit the assignment"
]

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


def get_predicate_from_nl(list_of_premises_nl):
    prompt = f"""<s>[INST]
### Task: From a list of **natural language premises**, extract **complete predicate phrases** (main actions or properties) that capture the core meaning of each sentence.

You are given:
- A list of natural language premises, each describing facts, rules, or knowledge in a domain.
- Your task is to extract **predicate phrases** from each premise — these are the main actions or properties (including passive constructions) expressed in each sentence.

---

### Extraction Rules (Apply to each premise separately):

1. **Extract only the core predicate phrase**:
   - Focus on **main actions or properties** (typically verbs and their complements/modifiers).
   - **Exclude subjects** (e.g., "students", "people") and only extract the **predicate phrase**.
   - Do **not** remove essential complements, prepositions, or passive markers.
     - ✅ Correct: "submitted before the deadline", "signed by the teacher"
     - ❌ Incorrect: "submitted", "signed"

2. **Remove auxiliary and modal verbs**:
   - Delete: am/is/are/was/were/be/do/does/did, including “not”
   - Delete: must/should/shall/will/would/can/could/may/might/ought to
   - Keep **only the lexical/main verb** and its **associated phrase**.

3. **Preserve prepositional and passive structures**:
   - Retain any parts that complete the verb’s meaning (e.g., "evaluated by the committee").

4. **Split compound predicates**:
   - If multiple predicates are joined by "and"/"or", **split them** into separate predicate phrases.
   - When a modifier applies to multiple predicates, **replicate it** for each.

5. **Handle conditional or negative constructions carefully**:
   - Extract the **core action separately** from conditions.
   - Do not merge conditional clauses unless they affect the **semantic meaning** of the predicate.

---

### Output Format:
- Return a **single list** of predicate phrases extracted from **all premises**.
- Use this exact format:  
  **List of predicate: ["predicate phrase 1", "predicate phrase 2", ..., "predicate phrase N"]**
- Do **not repeat** the premises or provide any explanation.
- **Only output this list**, and ensure it contains **no duplicates** unless a phrase appears with distinct meaning.

---

### Input:
- List of natural language premises:
{list_of_premises_nl}
[/INST]
Output: </s>"""
    return prompt



