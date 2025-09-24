import json
from torch.utils.data import Dataset
from typing import Dict, Any


class DPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer, template, max_length: int):
        self.data = self.load_dpo_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        self._validate_data()

    def load_dpo_data(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"DPO data file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"DPO data file JSON format error: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading DPO data file: {e}")

    def _validate_data(self):
        required_fields = ["prompt", "chosen", "rejected"]
        for i, item in enumerate(self.data):
            if not isinstance(item, dict):
                raise ValueError(f"Data {i} is not a dictionary")
            
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Data {i} is missing required field: {field}")
                
                if not isinstance(item[field], str):
                    raise ValueError(f"Data {i} field {field} is not a string")
                
                if not item[field].strip():
                    raise ValueError(f"Data {i} field {field} is empty")
        
        print("DPO data format validation passed")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        return {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }