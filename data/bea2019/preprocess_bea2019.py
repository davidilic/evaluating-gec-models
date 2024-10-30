import json
import nltk

def process_edits(text: str, edits: list) -> str:
    """
    Apply edits to the text to create the corrected version.
    """
    chars = list(text)
    
    all_individual_edits = []
    for edit_group in edits:
        start_pos = edit_group[0]
        for edit in edit_group[1]:
            all_individual_edits.append((start_pos + edit[0], start_pos + edit[1], edit[2]))
    
    all_individual_edits.sort(key=lambda x: x[0], reverse=True)
    
    for start, end, replacement in all_individual_edits:
        if replacement is None:
            chars[start:end] = []
        else:
            chars[start:end] = list(replacement)

    return ''.join(chars)

def create_dataset(input_file_path: str, output_file_path: str):
    """
    Takes in the raw data file and turns it into a jsonl file with two fields: source and target.
    """
    data = []

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        for entry in data:
            original_text = entry['text']
            corrected_text = process_edits(entry['text'], entry['edits'])
            
            output_entry = {
                'source': original_text.strip(),
                'target': corrected_text.strip(),
                "language": "english"
            }
            
            out_file.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    input_file = "data/bea2019/bea2019_dev_raw.jsonl"
    output_file = "data/bea2019/bea2019_processed.jsonl"

    create_dataset(input_file, output_file)