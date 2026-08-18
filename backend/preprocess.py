"""
preprocess.py — Dataset loading & medical information extraction utilities.
Shared across training scripts and inference pipeline.
"""

import os
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder


# ── Medical information extraction ──────────────────────────────────────────

ORGANS = {
    'lung':     ['lung', 'pulmonary', 'bronchial', 'respiratory', 'chest', 'thorax', 'lobe'],
    'brain':    ['brain', 'cerebral', 'cranial', 'intracranial', 'head', 'neurological'],
    'heart':    ['heart', 'cardiac', 'coronary', 'myocardial', 'cardio', 'aorta'],
    'liver':    ['liver', 'hepatic', 'hepat', 'biliary'],
    'kidney':   ['kidney', 'renal', 'neph', 'urinary'],
    'breast':   ['breast', 'mammary', 'mammograph'],
    'skin':     ['skin', 'dermal', 'cutaneous', 'melanocyte'],
    'bone':     ['bone', 'skeletal', 'osseous', 'fracture', 'spine', 'vertebr'],
    'colon':    ['colon', 'colonic', 'colorectal', 'bowel', 'intestin'],
    'prostate': ['prostate', 'prostatic'],
    'eye':      ['eye', 'retina', 'optic', 'ocular', 'fundus'],
}

DIAGNOSES = {
    'normal':      ['normal', 'healthy', 'unremarkable', 'no abnormality', 'no evidence', 'negative'],
    'benign':      ['benign', 'non-cancerous', 'non-malignant', 'cyst', 'fibroadenoma'],
    'malignant':   ['malignant', 'cancer', 'carcinoma', 'tumor', 'neoplasm', 'metast'],
    'abnormal':    ['abnormal', 'abnormality', 'pathological', 'disease', 'lesion'],
    'infection':   ['infection', 'infectious', 'inflammation', 'inflammatory', 'pneumonia'],
    'fracture':    ['fracture', 'broken', 'break'],
    'hemorrhage':  ['hemorrhage', 'bleeding', 'hematoma'],
    'edema':       ['edema', 'swelling', 'fluid', 'effusion'],
}


def extract_medical_info(text: str) -> tuple[str, str]:
    """Extract organ and diagnosis from free-text answer."""
    text_lower = str(text).lower()

    detected_organ = 'unknown'
    for organ, keywords in ORGANS.items():
        if any(kw in text_lower for kw in keywords):
            detected_organ = organ
            break

    detected_diagnosis = 'unknown'
    for diagnosis, keywords in DIAGNOSES.items():
        if any(kw in text_lower for kw in keywords):
            detected_diagnosis = diagnosis
            break

    return detected_organ, detected_diagnosis


# ── Dataset loading ─────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and prepare the VQA dataset."""
    df = pd.read_csv(csv_path)

    # Normalise column names
    if 'Questions' in df.columns:
        df = df.rename(columns={'Questions': 'question', 'Answers': 'answer'})

    # Extract medical info
    organs, diagnoses = [], []
    for answer in df['answer']:
        organ, diagnosis = extract_medical_info(answer)
        organs.append(organ)
        diagnoses.append(diagnosis)

    df['organ'] = organs
    df['diagnosis'] = diagnoses
    df['combined_label'] = df['organ'] + '_' + df['diagnosis']

    return df


def build_encoders(df: pd.DataFrame) -> dict:
    """Build label encoders for organ, diagnosis and combined labels."""
    organ_enc = LabelEncoder().fit(df['organ'])
    diagnosis_enc = LabelEncoder().fit(df['diagnosis'])
    combined_enc = LabelEncoder().fit(df['combined_label'])

    return {
        'organ': organ_enc,
        'diagnosis': diagnosis_enc,
        'combined': combined_enc,
    }


# ── Text utilities ──────────────────────────────────────────────────────────

def clean_question(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text
