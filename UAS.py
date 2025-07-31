import os
import json
import csv
import base64
import requests
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
import difflib
from PIL import Image
import time
import re

@dataclass
class OCRResult:
    image_path: str
    ground_truth: str
    prediction: str
    cer_score: float

class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/v1/chat/completions"

    def encode_image_to_base64(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def predict_license_plate(self, image_path: str, model_name: str = "c") -> str:
        try:
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return ""

            prompt_text = (
                "What	is	the	license	plate	number	shown	in	this	image?	Respond	only	with	the	plate	number "
            )

            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }

            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=50000
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result['choices'][0]['message']['content'].strip()
                return self.clean_prediction(prediction)
            else:
                print(f"API error {response.status_code}: {response.text}")
                return ""
        except Exception as e:
            print(f"Prediction error: {e}")
            return ""

    def clean_prediction(self, prediction: str) -> str:
        prediction = prediction.upper().strip()
        prediction = re.sub(r'[^A-Z0-9]', '', prediction)

        pattern = r'[A-Z]{1,2}\d{1,4}[A-Z]{1,3}'
        match = re.search(pattern, prediction)
        if match:
            return match.group()

        return prediction

class CERCalculator:
    @staticmethod
    def calculate_cer(ground_truth: str, prediction: str) -> float:
        if not ground_truth:
            return 1.0 if prediction else 0.0

        sm = difflib.SequenceMatcher(None, ground_truth.upper(), prediction.upper())
        substitutions = deletions = insertions = 0

        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == 'replace':
                substitutions += max(i2 - i1, j2 - j1)
            elif op == 'delete':
                deletions += i2 - i1
            elif op == 'insert':
                insertions += j2 - j1

        total_errors = substitutions + deletions + insertions
        return total_errors / len(ground_truth)

    @staticmethod
    def calculate_detailed_cer(ground_truth: str, prediction: str) -> Dict:
        if not ground_truth:
            return {
                'cer': 1.0 if prediction else 0.0,
                'substitutions': 0,
                'deletions': 0,
                'insertions': len(prediction),
                'total_errors': len(prediction),
                'ground_truth_length': 0
            }

        sm = difflib.SequenceMatcher(None, ground_truth.upper(), prediction.upper())
        substitutions = deletions = insertions = 0

        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == 'replace':
                substitutions += max(i2 - i1, j2 - j1)
            elif op == 'delete':
                deletions += i2 - i1
            elif op == 'insert':
                insertions += j2 - j1

        total_errors = substitutions + deletions + insertions
        return {
            'cer': total_errors / len(ground_truth),
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'total_errors': total_errors,
            'ground_truth_length': len(ground_truth)
        }

class LicensePlateOCR:
    def __init__(self, lmstudio_url: str = "http://localhost:1234", model_name: str = "llava-v1.5-7b-gpt4ocr-hf", request_delay=0.3):
        self.client = LMStudioClient(lmstudio_url)
        self.model_name = model_name
        self.cer_calculator = CERCalculator()
        self.results: List[OCRResult] = []
        self.request_delay = request_delay

    def load_dataset(self, dataset_path: str, ground_truth_file: str = None) -> Dict[str, str]:
        gt = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            try:
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        gt[row['image']] = row['ground_truth']
            except Exception as e:
                print(f"Failed to read ground truth file: {e}")
        return gt

    def process_single_image(self, image_path: str, ground_truth: str = "") -> OCRResult:
        print(f"Processing: {image_path}")
        prediction = self.client.predict_license_plate(image_path, self.model_name)
        cer_score = self.cer_calculator.calculate_cer(ground_truth, prediction)

        # Logging
        with open("log.txt", "a") as f:
            f.write(f"{os.path.basename(image_path)} | GT: {ground_truth} | Pred: {prediction} | CER: {cer_score:.4f}\n")

        print(f"GT: {ground_truth} | Pred: {prediction} | CER: {cer_score:.4f}")
        print("-" * 50)

        return OCRResult(image_path, ground_truth, prediction, cer_score)

    def process_dataset(self, dataset_path: str, ground_truth_file: str = None) -> List[OCRResult]:
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            return []

        gt_dict = self.load_dataset(dataset_path, ground_truth_file)
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in Path(dataset_path).iterdir() if f.suffix.lower() in valid_exts]

        print(f"Found {len(image_files)} image files")

        results = []
        for image_path in image_files:
            gt = gt_dict.get(image_path.name, "")
            try:
                result = self.process_single_image(str(image_path), gt)
                results.append(result)
                time.sleep(self.request_delay)
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue

        self.results = results
        return results

    def save_results_to_csv(self, output_file: str = "ocr_results.csv"):
        if not self.results:
            print("No results to save.")
            return

        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image', 'ground_truth', 'prediction', 'CER_score'])
                for r in self.results:
                    writer.writerow([os.path.basename(r.image_path), r.ground_truth, r.prediction, f"{r.cer_score:.4f}"])
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def calculate_overall_metrics(self) -> Dict:
        if not self.results:
            return {}

        total_cer = sum(r.cer_score for r in self.results)
        avg_cer = total_cer / len(self.results)
        correct = sum(1 for r in self.results if r.ground_truth.upper() == r.prediction.upper() and r.ground_truth != "")
        total_with_gt = sum(1 for r in self.results if r.ground_truth != "")
        accuracy = correct / total_with_gt if total_with_gt else 0.0

        subs = dels = ins = gt_len = 0
        for r in self.results:
            if r.ground_truth:
                d = self.cer_calculator.calculate_detailed_cer(r.ground_truth, r.prediction)
                subs += d['substitutions']
                dels += d['deletions']
                ins += d['insertions']
                gt_len += d['ground_truth_length']

        return {
            'total_images': len(self.results),
            'average_cer': avg_cer,
            'accuracy': accuracy,
            'correct_predictions': correct,
            'images_with_ground_truth': total_with_gt,
            'substitutions': subs,
            'deletions': dels,
            'insertions': ins,
            'total_gt_length': gt_len
        }

    def print_summary(self):
        m = self.calculate_overall_metrics()
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total Images: {m['total_images']}")
        print(f"With Ground Truth: {m['images_with_ground_truth']}")
        print(f"Correct Predictions: {m['correct_predictions']}")
        print(f"Accuracy: {m['accuracy']*100:.2f}%")
        print(f"Average CER: {m['average_cer']:.4f}")
        print(f"Substitutions: {m['substitutions']}")
        print(f"Deletions: {m['deletions']}")
        print(f"Insertions: {m['insertions']}")
        print("="*60)

def main():
    DATASET_PATH = r"C:\uas_vision\test"
    GROUND_TRUTH_FILE = r"C:\uas_vision\test\ground_truth.csv"
    OUTPUT_FILE = "ocr_results.csv"
    LMSTUDIO_URL = "http://localhost:1234"
    MODEL_NAME = "llava-v1.5-7b-gpt4ocr-hf"

    ocr = LicensePlateOCR(LMSTUDIO_URL, MODEL_NAME)

    print("Starting OCR...")
    results = ocr.process_dataset(DATASET_PATH, GROUND_TRUTH_FILE)

    if results:
        ocr.save_results_to_csv(OUTPUT_FILE)
        ocr.print_summary()
    else:
        print("No results. Check dataset or model.")

if __name__ == "__main__":
    main()
