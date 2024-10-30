from torch.utils.data import Dataset, DataLoader
from typing import Dict
import json


class GECDataset(Dataset):
    """Custom Dataset for GEC tasks"""

    def __init__(self, file_path: str):
        self.data = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]

        return {"source": item["source"], "target": item["target"], "language": item["language"]}


def load_gec_data(file_path: str):
    dataset = GECDataset(file_path)
    return DataLoader(dataset, collate_fn=lambda batch: batch[0])
