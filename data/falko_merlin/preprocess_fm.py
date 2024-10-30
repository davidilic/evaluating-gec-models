import json


def create_jsonl_from_parallel_files(src_file: str, tgt_file: str, output_file: str):
    """
    Convert parallel source and target files into a JSONL file with "source" and "target" fields.
    Each line from source and target files will be combined into a JSON object.
    """
    try:
        with open(src_file, 'r', encoding='utf-8') as src, \
             open(tgt_file, 'r', encoding='utf-8') as tgt, \
             open(output_file, 'w', encoding='utf-8') as out:
            
            for src_line, tgt_line in zip(src, tgt):
                src_text = src_line.strip()
                tgt_text = tgt_line.strip()
                
                if src_text and tgt_text:
                    entry = {
                        "source": src_text,
                        "target": tgt_text,
                        "language": "german"
                    }
                    out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    
    src_file = "data/falko_merlin/fm-test.src"
    tgt_file = "data/falko_merlin/fm-test.trg"
    output_file = "data/falko_merlin/fm_processed.jsonl"
    
    create_jsonl_from_parallel_files(src_file, tgt_file, output_file)