S# ==================== 1. INSTALL REQUIRED PACKAGES ====================
!pip install -q transformers torch torchvision pillow
!pip install -q timm  # For ConvNeXt
!pip install -q accelerate
!pip install -q sentencepiece

print("✅ Required packages installed")

# ==================== 2. IMPORTS ====================
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# BioBERT imports
from transformers import AutoTokenizer, AutoModel
import timm  # For ConvNeXt

print(f"PyTorch version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 3. LOAD AND PREPARE DATA ====================
print("\n📂 LOADING DATASET...")

dataset_path = '/content/drive/MyDrive/archive (2)'
csv_path = os.path.join(dataset_path, 'VQA_dataset.csv')

df = pd.read_csv(csv_path)
df = df.rename(columns={'Questions': 'question', 'Answers': 'answer'})

print(f"Dataset loaded: {df.shape}")

# Extract organ and diagnosis info
def extract_medical_info(answer):
    answer_lower = str(answer).lower()

    organs = {
        'lung': ['lung', 'pulmonary', 'bronchial', 'respiratory'],
        'brain': ['brain', 'cerebral', 'cranial', 'intracranial'],
        'heart': ['heart', 'cardiac', 'coronary', 'myocardial'],
        'liver': ['liver', 'hepatic', 'hepat'],
        'kidney': ['kidney', 'renal', 'neph'],
        'breast': ['breast', 'mammary'],
        'skin': ['skin', 'dermal', 'cutaneous'],
        'bone': ['bone', 'skeletal', 'osseous'],
        'colon': ['colon', 'colonic', 'colorectal'],
        'prostate': ['prostate', 'prostatic']
    }

    diagnoses = {
        'normal': ['normal', 'healthy', 'unremarkable', 'no abnormality'],
        'benign': ['benign', 'non-cancerous', 'non-malignant'],
        'malignant': ['malignant', 'cancer', 'carcinoma', 'tumor', 'neoplasm'],
        'abnormal': ['abnormal', 'abnormality', 'pathological', 'disease'],
        'infection': ['infection', 'infectious', 'inflammation', 'inflammatory'],
        'fracture': ['fracture', 'broken', 'break'],
        'hemorrhage': ['hemorrhage', 'bleeding', 'hematoma'],
        'edema': ['edema', 'swelling', 'fluid']
    }

    detected_organ = 'unknown'
    for organ, keywords in organs.items():
        if any(keyword in answer_lower for keyword in keywords):
            detected_organ = organ
            break

    detected_diagnosis = 'unknown'
    for diagnosis, keywords in diagnoses.items():
        if any(keyword in answer_lower for keyword in keywords):
            detected_diagnosis = diagnosis
            break

    return detected_organ, detected_diagnosis

# Apply extraction
organ_list, diagnosis_list = [], []
for answer in df['answer']:
    organ, diagnosis = extract_medical_info(answer)
    organ_list.append(organ)
    diagnosis_list.append(diagnosis)

df['organ'] = organ_list
df['diagnosis'] = diagnosis_list
df['combined_label'] = df['organ'] + '_' + df['diagnosis']

print(f"\n✅ Extracted medical information:")
print(f"  Organ classes: {df['organ'].nunique()}")
print(f"  Diagnosis classes: {df['diagnosis'].nunique()}")

# Encode labels
organ_encoder = LabelEncoder()
diagnosis_encoder = LabelEncoder()
combined_encoder = LabelEncoder()

df['organ_encoded'] = organ_encoder.fit_transform(df['organ'])
df['diagnosis_encoded'] = diagnosis_encoder.fit_transform(df['diagnosis'])
df['combined_encoded'] = combined_encoder.fit_transform(df['combined_label'])

organ_classes = organ_encoder.classes_
diagnosis_classes = diagnosis_encoder.classes_
combined_classes = combined_encoder.classes_

print(f"\n🏷️ Label encoding complete:")
print(f"  Organ classes: {len(organ_classes)}")
print(f"  Diagnosis classes: {len(diagnosis_classes)}")

# ==================== 4. CREATE DATASET CLASS ====================
class MedicalVQADataset(Dataset):
    def __init__(self, df, image_folder, tokenizer, transform=None, max_samples=None):
        self.df = df if max_samples is None else df.sample(max_samples, random_state=42)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_filename = row['image']
        img_paths = [
            os.path.join(self.image_folder, str(img_filename)),
            os.path.join(self.image_folder, 'images', str(img_filename)),
            os.path.join('/content/drive/MyDrive/archive (2)', str(img_filename)),
        ]

        img = None
        for img_path in img_paths:
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    break
                except:
                    continue

        if img is None:
            # Create synthetic image
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Tokenize question
        question = str(row['question'])
        tokenized = self.tokenizer(
            question,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=128
        )

        # Get labels
        organ_label = torch.tensor(row['organ_encoded'], dtype=torch.long)
        diagnosis_label = torch.tensor(row['diagnosis_encoded'], dtype=torch.long)
        combined_label = torch.tensor(row['combined_encoded'], dtype=torch.long)

        return {
            'image': img,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'token_type_ids': tokenized.get('token_type_ids', torch.zeros_like(tokenized['attention_mask'])).squeeze(0),
            'organ_label': organ_label,
            'diagnosis_label': diagnosis_label,
            'combined_label': combined_label,
            'question': question
        }

# ==================== 5. CREATE CONVNEXT + BIOBERT MODEL ====================
class ConvNeXtBioBERTMedicalVQA(nn.Module):
    def __init__(self, num_organs, num_diagnosis, num_combined,
                 convnext_model='convnext_base',
                 biobert_model_name='dmis-lab/biobert-v1.1'):
        super(ConvNeXtBioBERTMedicalVQA, self).__init__()

        print(f"\n🏗️ BUILDING ConvNeXt + BioBERT MODEL...")

        # Load BioBERT
        print(f"  Loading BioBERT ({biobert_model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
        self.biobert = AutoModel.from_pretrained(biobert_model_name)

        # Freeze BioBERT layers initially
        for param in self.biobert.parameters():
            param.requires_grad = False

        # Load ConvNeXt
        print(f"  Loading {convnext_model}...")
        self.convnext = timm.create_model(
            convnext_model,
            pretrained=True,
            num_classes=0,  # Remove classification head
            features_only=False
        )

        # Freeze ConvNeXt initially
        for param in self.convnext.parameters():
            param.requires_grad = False

        # Get feature dimensions
        convnext_features = 1024 if 'base' in convnext_model else 768
        biobert_hidden_size = self.biobert.config.hidden_size

        print(f"  ConvNeXt features: {convnext_features}")
        print(f"  BioBERT hidden size: {biobert_hidden_size}")

        # Feature projection layers
        self.visual_projection = nn.Sequential(
            nn.Linear(convnext_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, biobert_hidden_size),
            nn.LayerNorm(biobert_hidden_size),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Attention-based fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=biobert_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(biobert_hidden_size)

        # Multi-modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(biobert_hidden_size * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Multi-task classification heads
        # Organ classification head
        self.organ_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_organs)
        )

        # Diagnosis classification head
        self.diagnosis_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_diagnosis)
        )

        # Combined classification head
        self.combined_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_combined)
        )

        # Layer scaling (ConvNeXt style)
        self.layer_scale_visual = nn.Parameter(torch.ones(1, biobert_hidden_size) * 1e-6)
        self.layer_scale_text = nn.Parameter(torch.ones(1, biobert_hidden_size) * 1e-6)

        print("✅ Model initialized successfully!")
        print(f"   Visual encoder: {convnext_model}")
        print(f"   Language model: BioBERT (biomedical domain)")
        print(f"   Fusion: Cross-attention + projection")
        print(f"   Multi-task heads: Organ + Diagnosis + Combined")

    def forward(self, images, input_ids, attention_mask, token_type_ids=None):
        batch_size = images.size(0)

        # Extract visual features using ConvNeXt
        visual_features = self.convnext.forward_features(images)  # [batch, features, H, W]
        visual_features = visual_features.mean(dim=[2, 3])  # Global average pooling

        # Project visual features to BioBERT dimension
        visual_features = self.visual_projection(visual_features)  # [batch, hidden_size]
        visual_features = visual_features.unsqueeze(1)  # [batch, 1, hidden_size]

        # Get BioBERT features
        biobert_outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        # Use the [CLS] token for text representation
        text_features = biobert_outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        text_features = text_features.unsqueeze(1)  # [batch, 1, hidden_size]

        # Cross-attention between visual and text features
        # Visual as query, Text as key/value
        attended_visual, _ = self.cross_attention(
            query=visual_features,
            key=text_features,
            value=text_features
        )

        # Text as query, Visual as key/value
        attended_text, _ = self.cross_attention(
            query=text_features,
            key=visual_features,
            value=visual_features
        )

        # Apply layer scaling (ConvNeXt style)
        attended_visual = attended_visual * self.layer_scale_visual
        attended_text = attended_text * self.layer_scale_text

        # Normalize
        attended_visual = self.attention_norm(attended_visual)
        attended_text = self.attention_norm(attended_text)

        # Concatenate and fuse
        attended_visual = attended_visual.squeeze(1)  # [batch, hidden_size]
        attended_text = attended_text.squeeze(1)  # [batch, hidden_size]

        fused_features = torch.cat([attended_visual, attended_text], dim=1)
        fused = self.fusion(fused_features)

        # Multi-task predictions
        organ_logits = self.organ_head(fused)
        diagnosis_logits = self.diagnosis_head(fused)
        combined_logits = self.combined_head(fused)

        return organ_logits, diagnosis_logits, combined_logits

# ==================== 6. SETUP DATALOADERS ====================
print("\n📊 SETTING UP DATA PIPELINE...")

# Image transforms for ConvNeXt
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize BioBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

# Split data
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Data split:")
print(f"  Training: {len(train_df)} samples")
print(f"  Validation: {len(val_df)} samples")
print(f"  Test: {len(test_df)} samples")

# Create datasets
image_folder = '/content/drive/MyDrive/archive (2)/QA_VLM_MED/images/images'

train_dataset = MedicalVQADataset(train_df, image_folder, tokenizer, transform, max_samples=1500)
val_dataset = MedicalVQADataset(val_df, image_folder, tokenizer, transform, max_samples=300)
test_dataset = MedicalVQADataset(test_df, image_folder, tokenizer, transform, max_samples=300)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

print(f"\n✅ Data loaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ==================== 7. INITIALIZE MODEL ====================
print("\n🎯 INITIALIZING ConvNeXt + BioBERT MODEL...")

num_organs = len(organ_classes)
num_diagnosis = len(diagnosis_classes)
num_combined = len(combined_classes)

model = ConvNeXtBioBERTMedicalVQA(
    num_organs=num_organs,
    num_diagnosis=num_diagnosis,
    num_combined=num_combined,
    convnext_model='convnext_base',
    biobert_model_name='dmis-lab/biobert-v1.1'
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Model parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")
print(f"  Frozen: {total_params - trainable_params:,}")

# ==================== 8. TRAINING SETUP ====================
print("\n🚀 SETTING UP TRAINING...")

# Loss functions
criterion_organ = nn.CrossEntropyLoss()
criterion_diagnosis = nn.CrossEntropyLoss()
criterion_combined = nn.CrossEntropyLoss()

# Optimizer with layer-wise learning rate decay
optimizer = optim.AdamW([
    {'params': model.convnext.parameters(), 'lr': 1e-5},
    {'params': model.biobert.parameters(), 'lr': 2e-5},
    {'params': model.visual_projection.parameters(), 'lr': 3e-4},
    {'params': model.cross_attention.parameters(), 'lr': 3e-4},
    {'params': model.fusion.parameters(), 'lr': 3e-4},
    {'params': model.organ_head.parameters(), 'lr': 3e-4},
    {'params': model.diagnosis_head.parameters(), 'lr': 3e-4},
    {'params': model.combined_head.parameters(), 'lr': 3e-4},
], weight_decay=0.01)

# Cosine annealing scheduler with warmup
def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training steps
num_training_steps = len(train_loader) * 10  # 10 epochs
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

# Gradient accumulation for larger effective batch size
gradient_accumulation_steps = 4

# Training function
def train_epoch(model, dataloader, optimizer, device, scheduler=None, grad_accum_steps=4):
    model.train()
    total_loss = 0
    organ_correct = 0
    diagnosis_correct = 0
    combined_correct = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        organ_labels = batch['organ_label'].to(device)
        diagnosis_labels = batch['diagnosis_label'].to(device)
        combined_labels = batch['combined_label'].to(device)

        # Forward pass
        organ_logits, diagnosis_logits, combined_logits = model(
            images, input_ids, attention_mask, token_type_ids
        )

        # Calculate losses
        loss_organ = criterion_organ(organ_logits, organ_labels)
        loss_diagnosis = criterion_diagnosis(diagnosis_logits, diagnosis_labels)
        loss_combined = criterion_combined(combined_logits, combined_labels)

        # Weighted multi-task loss
        total_loss_batch = 0.4 * loss_organ + 0.3 * loss_diagnosis + 0.3 * loss_combined

        # Scale loss for gradient accumulation
        total_loss_batch = total_loss_batch / grad_accum_steps
        total_loss_batch.backward()

        # Calculate accuracy
        organ_preds = organ_logits.argmax(dim=1)
        diagnosis_preds = diagnosis_logits.argmax(dim=1)
        combined_preds = combined_logits.argmax(dim=1)

        organ_correct += (organ_preds == organ_labels).sum().item()
        diagnosis_correct += (diagnosis_preds == diagnosis_labels).sum().item()
        combined_correct += (combined_preds == combined_labels).sum().item()

        total_samples += organ_labels.size(0)
        total_loss += total_loss_batch.item() * grad_accum_steps  # Scale back for logging

        # Gradient accumulation step
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        # Print progress
        if (batch_idx + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {total_loss_batch.item()*grad_accum_steps:.4f}, LR: {current_lr:.2e}")

    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    organ_acc = organ_correct / total_samples
    diagnosis_acc = diagnosis_correct / total_samples
    combined_acc = combined_correct / total_samples

    return avg_loss, organ_acc, diagnosis_acc, combined_acc

# Validation function
def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    organ_correct = 0
    diagnosis_correct = 0
    combined_correct = 0
    total_samples = 0

    all_organ_preds = []
    all_organ_labels = []
    all_diagnosis_preds = []
    all_diagnosis_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            organ_labels = batch['organ_label'].to(device)
            diagnosis_labels = batch['diagnosis_label'].to(device)
            combined_labels = batch['combined_label'].to(device)

            # Forward pass
            organ_logits, diagnosis_logits, combined_logits = model(
                images, input_ids, attention_mask, token_type_ids
            )

            # Calculate losses
            loss_organ = criterion_organ(organ_logits, organ_labels)
            loss_diagnosis = criterion_diagnosis(diagnosis_logits, diagnosis_labels)
            loss_combined = criterion_combined(combined_logits, combined_labels)

            total_loss_batch = 0.4 * loss_organ + 0.3 * loss_diagnosis + 0.3 * loss_combined
            total_loss += total_loss_batch.item()

            # Calculate predictions
            organ_preds = organ_logits.argmax(dim=1)
            diagnosis_preds = diagnosis_logits.argmax(dim=1)
            combined_preds = combined_logits.argmax(dim=1)

            organ_correct += (organ_preds == organ_labels).sum().item()
            diagnosis_correct += (diagnosis_preds == diagnosis_labels).sum().item()
            combined_correct += (combined_preds == combined_labels).sum().item()

            total_samples += organ_labels.size(0)

            # Store for metrics
            all_organ_preds.extend(organ_preds.cpu().numpy())
            all_organ_labels.extend(organ_labels.cpu().numpy())
            all_diagnosis_preds.extend(diagnosis_preds.cpu().numpy())
            all_diagnosis_labels.extend(diagnosis_labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    organ_acc = organ_correct / total_samples
    diagnosis_acc = diagnosis_correct / total_samples
    combined_acc = combined_correct / total_samples

    return (avg_loss, organ_acc, diagnosis_acc, combined_acc,
            all_organ_preds, all_organ_labels, all_diagnosis_preds, all_diagnosis_labels)

# ==================== 9. TRAINING LOOP ====================
print("\n" + "="*60)
print("STARTING TRAINING - ConvNeXt + BioBERT")
print("="*60)

num_epochs = 10
best_val_loss = float('inf')
patience = 5
patience_counter = 0

train_history = {
    'loss': [], 'organ_acc': [], 'diagnosis_acc': [], 'combined_acc': [],
    'val_loss': [], 'val_organ_acc': [], 'val_diagnosis_acc': [], 'val_combined_acc': [],
    'learning_rate': []
}

for epoch in range(num_epochs):
    print(f"\n📊 EPOCH {epoch+1}/{num_epochs}")
    print("-" * 40)

    # Training
    train_loss, train_organ_acc, train_diagnosis_acc, train_combined_acc = train_epoch(
        model, train_loader, optimizer, device, scheduler, gradient_accumulation_steps
    )

    # Validation
    (val_loss, val_organ_acc, val_diagnosis_acc, val_combined_acc,
     val_organ_preds, val_organ_labels, val_diagnosis_preds, val_diagnosis_labels) = validate_epoch(
        model, val_loader, device
    )

    # Store history
    train_history['loss'].append(train_loss)
    train_history['organ_acc'].append(train_organ_acc)
    train_history['diagnosis_acc'].append(train_diagnosis_acc)
    train_history['combined_acc'].append(train_combined_acc)

    train_history['val_loss'].append(val_loss)
    train_history['val_organ_acc'].append(val_organ_acc)
    train_history['val_diagnosis_acc'].append(val_diagnosis_acc)
    train_history['val_combined_acc'].append(val_combined_acc)

    # Store learning rate
    current_lr = optimizer.param_groups[0]['lr']
    train_history['learning_rate'].append(current_lr)

    # Print epoch results
    print(f"Train - Loss: {train_loss:.4f}, Organ Acc: {train_organ_acc:.4f}, Diag Acc: {train_diagnosis_acc:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Organ Acc: {val_organ_acc:.4f}, Diag Acc: {val_diagnosis_acc:.4f}")
    print(f"Learning Rate: {current_lr:.2e}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_organ_acc': val_organ_acc,
            'val_diagnosis_acc': val_diagnosis_acc,
            'organ_encoder': organ_encoder,
            'diagnosis_encoder': diagnosis_encoder,
        }, '/content/best_convnext_biobert_model.pth')
        print("✅ Saved best model")
    else:
        patience_counter += 1
        print(f"⚠️ No improvement for {patience_counter} epoch(s)")

        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered")
            break

print("\n✅ Training completed!")

# ==================== 10. VISUALIZE TRAINING HISTORY ====================
print("\n📈 VISUALIZING TRAINING HISTORY...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss
axes[0, 0].plot(train_history['loss'], label='Train Loss', marker='o', linewidth=2)
axes[0, 0].plot(train_history['val_loss'], label='Val Loss', marker='s', linewidth=2)
axes[0, 0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Organ Accuracy
axes[0, 1].plot(train_history['organ_acc'], label='Train Organ Acc', marker='o', linewidth=2)
axes[0, 1].plot(train_history['val_organ_acc'], label='Val Organ Acc', marker='s', linewidth=2)
axes[0, 1].set_title('Organ Classification Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Diagnosis Accuracy
axes[0, 2].plot(train_history['diagnosis_acc'], label='Train Diagnosis Acc', marker='o', linewidth=2)
axes[0, 2].plot(train_history['val_diagnosis_acc'], label='Val Diagnosis Acc', marker='s', linewidth=2)
axes[0, 2].set_title('Diagnosis Classification Accuracy', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Accuracy')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Combined Accuracy
axes[1, 0].plot(train_history['combined_acc'], label='Train Combined Acc', marker='o', linewidth=2)
axes[1, 0].plot(train_history['val_combined_acc'], label='Val Combined Acc', marker='s', linewidth=2)
axes[1, 0].set_title('Combined Classification Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(train_history['learning_rate'], marker='o', linewidth=2, color='purple')
axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].grid(True, alpha=0.3)

# Accuracy Comparison
axes[1, 2].plot(train_history['organ_acc'], label='Organ Acc', marker='o', linewidth=2)
axes[1, 2].plot(train_history['diagnosis_acc'], label='Diagnosis Acc', marker='s', linewidth=2)
axes[1, 2].plot(train_history['combined_acc'], label='Combined Acc', marker='^', linewidth=2)
axes[1, 2].set_title('Accuracy Comparison (Train)', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Accuracy')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('ConvNeXt + BioBERT Training History', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/content/convnext_biobert_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Training history saved as 'convnext_biobert_training_history.png'")

# ==================== 11. TEST EVALUATION ====================
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

# Load best model
checkpoint = torch.load('/content/best_convnext_biobert_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

print("✅ Loaded best model for testing")

# Evaluate on test set
(test_loss, test_organ_acc, test_diagnosis_acc, test_combined_acc,
 test_organ_preds, test_organ_labels, test_diagnosis_preds, test_diagnosis_labels) = validate_epoch(
    model, test_loader, device
)

print(f"\n📊 TEST RESULTS:")
print(f"  Organ Accuracy:     {test_organ_acc:.4f}")
print(f"  Diagnosis Accuracy: {test_diagnosis_acc:.4f}")
print(f"  Combined Accuracy:  {test_combined_acc:.4f}")
print(f"  Test Loss:          {test_loss:.4f}")

# ==================== 12. CONFUSION MATRICES ====================
print("\n🎯 CREATING CONFUSION MATRICES...")

# Get unique classes in test set
unique_organs_test = np.unique(test_organ_labels)
unique_diagnoses_test = np.unique(test_diagnosis_labels)

organ_classes_test = [organ_classes[i] for i in unique_organs_test]
diagnosis_classes_test = [diagnosis_classes[i] for i in unique_diagnoses_test]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Organ Confusion Matrix
organ_cm = confusion_matrix(test_organ_labels, test_organ_preds, labels=unique_organs_test)
organ_cm_normalized = organ_cm.astype('float') / organ_cm.sum(axis=1)[:, np.newaxis]
organ_cm_normalized = np.nan_to_num(organ_cm_normalized)

sns.heatmap(organ_cm_normalized, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[0], cbar_kws={'label': 'Normalized Count'},
            xticklabels=[c[:10] for c in organ_classes_test],
            yticklabels=[c[:10] for c in organ_classes_test])
axes[0].set_title('Organ Classification - ConvNeXt + BioBERT', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Organ', fontsize=10)
# ==================== 12. CONFUSION MATRICES (继续) ====================
axes[0].set_ylabel('True Organ', fontsize=10)
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# Diagnosis Confusion Matrix
diagnosis_cm = confusion_matrix(test_diagnosis_labels, test_diagnosis_preds, labels=unique_diagnoses_test)
diagnosis_cm_normalized = diagnosis_cm.astype('float') / diagnosis_cm.sum(axis=1)[:, np.newaxis]
diagnosis_cm_normalized = np.nan_to_num(diagnosis_cm_normalized)

sns.heatmap(diagnosis_cm_normalized, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1], cbar_kws={'label': 'Normalized Count'},
            xticklabels=[c[:15] for c in diagnosis_classes_test],
            yticklabels=[c[:15] for c in diagnosis_classes_test])
axes[1].set_title('Diagnosis Classification - ConvNeXt + BioBERT', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Diagnosis', fontsize=10)
axes[1].set_ylabel('True Diagnosis', fontsize=10)
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

plt.suptitle('Test Set Performance - ConvNeXt + BioBERT', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/content/convnext_biobert_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Confusion matrices saved as 'convnext_biobert_confusion_matrices.png'")

# ==================== 13. CLASSIFICATION REPORTS ====================
print("\n📋 CLASSIFICATION REPORTS")
print("="*60)

print("\n📍 ORGAN CLASSIFICATION REPORT:")
organ_report = classification_report(
    test_organ_labels,
    test_organ_preds,
    target_names=[organ_classes[i] for i in unique_organs_test],
    digits=4
)
print(organ_report)

print("\n🏥 DIAGNOSIS CLASSIFICATION REPORT:")
diagnosis_report = classification_report(
    test_diagnosis_labels,
    test_diagnosis_preds,
    target_names=[diagnosis_classes[i] for i in unique_diagnoses_test],
    digits=4
)
print(diagnosis_report)

# ==================== 14. SAMPLE PREDICTIONS ====================
print("\n🔍 SAMPLE PREDICTIONS FROM TEST SET")
print("="*60)

def show_sample_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    samples = []

    with torch.no_grad():
        for batch in dataloader:
            if len(samples) >= num_samples:
                break

            # Get one batch
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            organ_labels = batch['organ_label']
            diagnosis_labels = batch['diagnosis_label']
            questions = batch['question']

            # Get predictions
            organ_logits, diagnosis_logits, _ = model(images, input_ids, attention_mask, token_type_ids)
            organ_preds = organ_logits.argmax(dim=1).cpu().numpy()
            diagnosis_preds = diagnosis_logits.argmax(dim=1).cpu().numpy()

            # Collect samples
            for i in range(min(len(questions), num_samples - len(samples))):
                sample = {
                    'question': questions[i],
                    'true_organ': organ_classes[organ_labels[i].item()],
                    'pred_organ': organ_classes[organ_preds[i]],
                    'true_diagnosis': diagnosis_classes[diagnosis_labels[i].item()],
                    'pred_diagnosis': diagnosis_classes[diagnosis_preds[i]],
                    'organ_correct': organ_labels[i].item() == organ_preds[i],
                    'diagnosis_correct': diagnosis_labels[i].item() == diagnosis_preds[i]
                }
                samples.append(sample)

    return samples

# Get sample predictions
sample_predictions = show_sample_predictions(model, test_loader, device, num_samples=8)

# Display samples
for i, sample in enumerate(sample_predictions):
    print(f"\n📌 Sample {i+1}:")
    print(f"   Question: {sample['question'][:100]}...")
    print(f"   True Organ: {sample['true_organ']:15} → Predicted: {sample['pred_organ']:15} {'✅' if sample['organ_correct'] else '❌'}")
    print(f"   True Diagnosis: {sample['true_diagnosis']:15} → Predicted: {sample['pred_diagnosis']:15} {'✅' if sample['diagnosis_correct'] else '❌'}")
    print(f"   {'CORRECT' if sample['organ_correct'] and sample['diagnosis_correct'] else 'PARTIALLY CORRECT' if sample['organ_correct'] or sample['diagnosis_correct'] else 'INCORRECT'}")

# ==================== 15. ERROR ANALYSIS ====================
print("\n🔧 ERROR ANALYSIS")
print("="*60)

# Find misclassified samples
def analyze_errors(model, dataloader, device):
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            organ_labels = batch['organ_label'].cpu().numpy()
            diagnosis_labels = batch['diagnosis_label'].cpu().numpy()
            questions = batch['question']

            # Get predictions
            organ_logits, diagnosis_logits, _ = model(images, input_ids, attention_mask, token_type_ids)
            organ_preds = organ_logits.argmax(dim=1).cpu().numpy()
            diagnosis_preds = diagnosis_logits.argmax(dim=1).cpu().numpy()

            # Find errors
            for i in range(len(organ_labels)):
                if organ_labels[i] != organ_preds[i] or diagnosis_labels[i] != diagnosis_preds[i]:
                    error = {
                        'question': questions[i],
                        'true_organ': organ_classes[organ_labels[i]],
                        'pred_organ': organ_classes[organ_preds[i]],
                        'true_diagnosis': diagnosis_classes[diagnosis_labels[i]],
                        'pred_diagnosis': diagnosis_classes[diagnosis_preds[i]],
                        'organ_error': organ_labels[i] != organ_preds[i],
                        'diagnosis_error': diagnosis_labels[i] != diagnosis_preds[i]
                    }
                    errors.append(error)

    return errors

# Analyze errors
errors = analyze_errors(model, test_loader, device)

print(f"\n📊 Error Statistics:")
print(f"  Total test samples: {len(test_dataset)}")
print(f"  Samples with errors: {len(errors)}")
print(f"  Overall accuracy: {(1 - len(errors)/len(test_dataset))*100:.2f}%")

if errors:
    # Categorize errors
    organ_only_errors = sum(1 for e in errors if e['organ_error'] and not e['diagnosis_error'])
    diagnosis_only_errors = sum(1 for e in errors if e['diagnosis_error'] and not e['organ_error'])
    both_errors = sum(1 for e in errors if e['organ_error'] and e['diagnosis_error'])

    print(f"\n📈 Error Breakdown:")
    print(f"  Organ-only errors: {organ_only_errors} ({organ_only_errors/len(errors)*100:.1f}%)")
    print(f"  Diagnosis-only errors: {diagnosis_only_errors} ({diagnosis_only_errors/len(errors)*100:.1f}%)")
    print(f"  Both organ and diagnosis errors: {both_errors} ({both_errors/len(errors)*100:.1f}%)")

    # Show most common error patterns
    print(f"\n🔍 Top 5 Error Patterns:")
    error_patterns = {}
    for error in errors:
        pattern = f"{error['true_organ']}→{error['pred_organ']} & {error['true_diagnosis']}→{error['pred_diagnosis']}"
        error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

    for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pattern}: {count} errors")

# ==================== 16. MODEL INFERENCE FUNCTION ====================
print("\n🤖 CREATING INFERENCE FUNCTION")
print("="*60)

def medical_vqa_inference(model, image_path, question, tokenizer, transform, device):
    """
    Perform inference on a single medical image and question
    """
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        print(f"⚠️ Could not load image: {image_path}")
        return None

    # Apply transforms
    if transform:
        image_tensor = transform(image).unsqueeze(0).to(device)

    # Tokenize question
    tokenized = tokenizer(
        question,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=128
    )

    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        organ_logits, diagnosis_logits, combined_logits = model(
            image_tensor, input_ids, attention_mask
        )

        organ_probs = torch.softmax(organ_logits, dim=1)
        diagnosis_probs = torch.softmax(diagnosis_logits, dim=1)

        organ_pred = organ_logits.argmax(dim=1).item()
        diagnosis_pred = diagnosis_logits.argmax(dim=1).item()

        organ_confidence = organ_probs[0, organ_pred].item()
        diagnosis_confidence = diagnosis_probs[0, diagnosis_pred].item()

    result = {
        'organ': organ_classes[organ_pred],
        'organ_confidence': organ_confidence,
        'diagnosis': diagnosis_classes[diagnosis_pred],
        'diagnosis_confidence': diagnosis_confidence,
        'combined': f"{organ_classes[organ_pred]}_{diagnosis_classes[diagnosis_pred]}",
        'question': question
    }

    return result

# Test inference with a sample
print("\n🧪 TEST INFERENCE WITH SAMPLE:")
test_image_path = '/content/drive/MyDrive/archive (2)/QA_VLM_MED/images/images/000001.jpg'
test_question = "What organ is shown in this image and what is the diagnosis?"

if os.path.exists(test_image_path):
    result = medical_vqa_inference(
        model=model,
        image_path=test_image_path,
        question=test_question,
        tokenizer=tokenizer,
        transform=transform,
        device=device
    )

    if result:
        print(f"  Question: {result['question']}")
        print(f"  Predicted Organ: {result['organ']} (confidence: {result['organ_confidence']:.3f})")
        print(f"  Predicted Diagnosis: {result['diagnosis']} (confidence: {result['diagnosis_confidence']:.3f})")
        print(f"  Combined: {result['combined']}")
else:
    print(f"⚠️ Test image not found at {test_image_path}")

# ==================== 17. MODEL SUMMARY AND SAVING ====================
print("\n💾 SAVING FINAL MODEL AND ARTIFACTS")
print("="*60)

# Save final model checkpoint
final_checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'train_history': train_history,
    'test_metrics': {
        'test_organ_acc': test_organ_acc,
        'test_diagnosis_acc': test_diagnosis_acc,
        'test_combined_acc': test_combined_acc,
        'test_loss': test_loss,
    },
    'encoders': {
        'organ_encoder': organ_encoder,
        'diagnosis_encoder': diagnosis_encoder,
        'combined_encoder': combined_encoder,
    },
    'classes': {
        'organ_classes': organ_classes.tolist(),
        'diagnosis_classes': diagnosis_classes.tolist(),
        'combined_classes': combined_classes.tolist(),
    },
    'model_config': {
        'convnext_model': 'convnext_base',
        'biobert_model': 'dmis-lab/biobert-v1.1',
        'num_organs': num_organs,
        'num_diagnosis': num_diagnosis,
        'num_combined': num_combined,
    }
}

torch.save(final_checkpoint, '/content/convnext_biobert_final_model.pth')
print("✅ Final model saved as 'convnext_biobert_final_model.pth'")

# Save encoders
import pickle
with open('/content/convnext_biobert_encoders.pkl', 'wb') as f:
    pickle.dump({
        'organ_encoder': organ_encoder,
        'diagnosis_encoder': diagnosis_encoder,
        'combined_encoder': combined_encoder,
    }, f)
print("✅ Encoders saved as 'convnext_biobert_encoders.pkl'")

# ==================== 18. PERFORMANCE SUMMARY ====================
print("\n" + "="*60)
print("FINAL PERFORMANCE SUMMARY - ConvNeXt + BioBERT")
print("="*60)

print(f"\n📊 MODEL ARCHITECTURE:")
print(f"  Visual Encoder: ConvNeXt Base (timm)")
print(f"  Language Model: BioBERT v1.1 (biomedical)")
print(f"  Fusion Mechanism: Cross-attention + Projection")
print(f"  Multi-task Heads: Organ + Diagnosis + Combined")

print(f"\n🎯 DATASET STATISTICS:")
print(f"  Total samples: {len(df)}")
print(f"  Training samples: {len(train_df)}")
print(f"  Validation samples: {len(val_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Organ classes: {num_organs}")
print(f"  Diagnosis classes: {num_diagnosis}")
print(f"  Combined classes: {num_combined}")

print(f"\n🏆 FINAL TEST PERFORMANCE:")
print(f"  Organ Classification Accuracy:     {test_organ_acc*100:.2f}%")
print(f"  Diagnosis Classification Accuracy: {test_diagnosis_acc*100:.2f}%")
print(f"  Combined Classification Accuracy:  {test_combined_acc*100:.2f}%")

print(f"\n📈 TRAINING METRICS:")
print(f"  Best Validation Loss: {best_val_loss:.4f}")
print(f"  Total Epochs Trained: {len(train_history['loss'])}")
print(f"  Final Learning Rate: {train_history['learning_rate'][-1]:.2e}")

print(f"\n💾 SAVED ARTIFACTS:")
print(f"  1. convnext_biobert_final_model.pth - Full model checkpoint")
print(f"  2. best_convnext_biobert_model.pth - Best model checkpoint")
print(f"  3. convnext_biobert_encoders.pkl - Label encoders")
print(f"  4. convnext_biobert_training_history.png - Training plots")
print(f"  5. convnext_biobert_confusion_matrices.png - Confusion matrices")

print(f"\n🚀 INFERENCE READY:")
print(f"  Use 'medical_vqa_inference()' function for predictions")
print(f"  Model supports: Organ + Diagnosis classification from medical images + questions")

print("\n" + "="*60)
print("🎉 MEDICAL VQA SYSTEM COMPLETED!")
print("="*60)