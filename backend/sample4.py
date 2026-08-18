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

# BLIP imports
from transformers import BlipProcessor, BlipModel, BlipForQuestionAnswering
import timm  # For EfficientNet

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
    def __init__(self, df, image_folder, processor, transform=None, max_samples=None):
        self.df = df if max_samples is None else df.sample(max_samples, random_state=42)
        self.image_folder = image_folder
        self.processor = processor
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
            img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))

        # Apply transforms if specified
        if self.transform:
            img = self.transform(img)

        # Get question
        question = str(row['question'])

        # Get labels
        organ_label = torch.tensor(row['organ_encoded'], dtype=torch.long)
        diagnosis_label = torch.tensor(row['diagnosis_encoded'], dtype=torch.long)
        combined_label = torch.tensor(row['combined_encoded'], dtype=torch.long)

        return {
            'image': img,
            'question': question,
            'organ_label': organ_label,
            'diagnosis_label': diagnosis_label,
            'combined_label': combined_label,
            'image_path': img_filename
        }

# ==================== 5. CREATE EFFICIENTNET + BLIP MODEL ====================
class EfficientNetBLIPMedicalVQA(nn.Module):
    def __init__(self, num_organs, num_diagnosis, num_combined,
                 efficientnet_model='efficientnet_b3',
                 blip_model_name="Salesforce/blip-vqa-base"):
        super(EfficientNetBLIPMedicalVQA, self).__init__()

        print(f"\n🏗️ BUILDING EfficientNet + BLIP MODEL...")

        # Load BLIP processor and model
        print("  Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained(blip_model_name)
        self.blip_model = BlipModel.from_pretrained(blip_model_name)

        # Freeze BLIP layers initially
        for param in self.blip_model.parameters():
            param.requires_grad = False

        # Load EfficientNet
        print(f"  Loading {efficientnet_model}...")
        self.efficientnet = timm.create_model(
            efficientnet_model,
            pretrained=True,
            num_classes=0  # Remove classification head
        )

        # Freeze EfficientNet initially
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Get feature dimensions
        efficientnet_features = 1536 if 'b3' in efficientnet_model else 1280
        blip_hidden_size = self.blip_model.config.vision_config.hidden_size

        print(f"  EfficientNet features: {efficientnet_features}")
        print(f"  BLIP hidden size: {blip_hidden_size}")

        # Feature fusion layers
        self.visual_projection = nn.Sequential(
            nn.Linear(efficientnet_features, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, blip_hidden_size),
            nn.LayerNorm(blip_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Multi-modal fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(blip_hidden_size * 2, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Multi-task classification heads
        # Organ classification head
        self.organ_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_organs)
        )

        # Diagnosis classification head
        self.diagnosis_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_diagnosis)
        )

        # Combined classification head
        self.combined_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_combined)
        )

        print("✅ Model initialized successfully!")
        print(f"   Visual encoder: {efficientnet_model}")
        print(f"   Vision-language model: BLIP")
        print(f"   Multi-task heads: Organ + Diagnosis + Combined")

    def forward(self, images, questions):
        batch_size = images.size(0)

        # Extract visual features using EfficientNet
        visual_features = self.efficientnet(images)  # [batch, features]

        # Project EfficientNet features to BLIP dimension
        visual_features = self.visual_projection(visual_features)
        visual_features = visual_features.unsqueeze(1)  # [batch, 1, hidden_size]

        # Process text with BLIP tokenizer
        inputs = self.processor(
            text=questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=40
        ).to(images.device)

        # Get BLIP features
        blip_outputs = self.blip_model(
            pixel_values=images,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )

        # Get visual and text embeddings from BLIP
        # Use the [CLS] token for text representation
        # The BlipModel output for text hidden states is in 'text_model_output.last_hidden_state'
        text_features = blip_outputs.text_model_output.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # Fuse visual and text features
        visual_features = visual_features.squeeze(1)  # [batch, hidden_size]
        fused_features = torch.cat([visual_features, text_features], dim=1)

        # Further fusion
        fused = self.fusion_layer(fused_features)

        # Multi-task predictions
        organ_logits = self.organ_head(fused)
        diagnosis_logits = self.diagnosis_head(fused)
        combined_logits = self.combined_head(fused)

        return organ_logits, diagnosis_logits, combined_logits

# ==================== 6. SETUP DATALOADERS ====================
print("\n📊 SETTING UP DATA PIPELINE...")

# Image transforms for EfficientNet
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # EfficientNet-B3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize BLIP processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# Split data
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Data split:")
print(f"  Training: {len(train_df)} samples")
print(f"  Validation: {len(val_df)} samples")
print(f"  Test: {len(test_df)} samples")

# Create datasets
image_folder = '/content/drive/MyDrive/archive (2)/QA_VLM_MED/images/images'

train_dataset = MedicalVQADataset(train_df, image_folder, processor, transform, max_samples=1500)
val_dataset = MedicalVQADataset(val_df, image_folder, processor, transform, max_samples=300)
test_dataset = MedicalVQADataset(test_df, image_folder, processor, transform, max_samples=300)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

print(f"\n✅ Data loaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ==================== 7. INITIALIZE MODEL ====================
print("\n🎯 INITIALIZING EfficientNet + BLIP MODEL...")

num_organs = len(organ_classes)
num_diagnosis = len(diagnosis_classes)
num_combined = len(combined_classes)

model = EfficientNetBLIPMedicalVQA(
    num_organs=num_organs,
    num_diagnosis=num_diagnosis,
    num_combined=num_combined,
    efficientnet_model='efficientnet_b3',
    blip_model_name="Salesforce/blip-vqa-base"
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

# Optimizer (only trainable parameters)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10,
    eta_min=1e-6
)

# Mixed precision training for efficiency
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Training function
def train_epoch(model, dataloader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    organ_correct = 0
    diagnosis_correct = 0
    combined_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        images = batch['image'].to(device)
        questions = batch['question']
        organ_labels = batch['organ_label'].to(device)
        diagnosis_labels = batch['diagnosis_label'].to(device)
        combined_labels = batch['combined_label'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Mixed precision training
        if scaler:
            with torch.cuda.amp.autocast():
                # Forward pass
                organ_logits, diagnosis_logits, combined_logits = model(images, questions)

                # Calculate losses
                loss_organ = criterion_organ(organ_logits, organ_labels)
                loss_diagnosis = criterion_diagnosis(diagnosis_logits, diagnosis_labels)
                loss_combined = criterion_combined(combined_logits, combined_labels)

                # Weighted multi-task loss
                total_loss_batch = 0.4 * loss_organ + 0.3 * loss_diagnosis + 0.3 * loss_combined

            # Backward pass with scaler
            scaler.scale(total_loss_batch).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without mixed precision
            organ_logits, diagnosis_logits, combined_logits = model(images, questions)

            loss_organ = criterion_organ(organ_logits, organ_labels)
            loss_diagnosis = criterion_diagnosis(diagnosis_logits, diagnosis_labels)
            loss_combined = criterion_combined(combined_logits, combined_labels)

            total_loss_batch = 0.4 * loss_organ + 0.3 * loss_diagnosis + 0.3 * loss_combined

            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Calculate accuracy
        organ_preds = organ_logits.argmax(dim=1)
        diagnosis_preds = diagnosis_logits.argmax(dim=1)
        combined_preds = combined_logits.argmax(dim=1)

        organ_correct += (organ_preds == organ_labels).sum().item()
        diagnosis_correct += (diagnosis_preds == diagnosis_labels).sum().item()
        combined_correct += (combined_preds == combined_labels).sum().item()

        total_samples += organ_labels.size(0)
        total_loss += total_loss_batch.item()

        # Print progress
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {total_loss_batch.item():.4f}")

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
            questions = batch['question']
            organ_labels = batch['organ_label'].to(device)
            diagnosis_labels = batch['diagnosis_label'].to(device)
            combined_labels = batch['combined_label'].to(device)

            # Forward pass
            organ_logits, diagnosis_logits, combined_logits = model(images, questions)

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
print("STARTING TRAINING - EfficientNet + BLIP")
print("="*60)

num_epochs = 20
best_val_loss = float('inf')
patience = 5
patience_counter = 0

train_history = {
    'loss': [], 'organ_acc': [], 'diagnosis_acc': [], 'combined_acc': [],
    'val_loss': [], 'val_organ_acc': [], 'val_diagnosis_acc': [], 'val_combined_acc': []
}

for epoch in range(num_epochs):
    print(f"\n📊 EPOCH {epoch+1}/{num_epochs}")
    print("-" * 40)

    # Training
    train_loss, train_organ_acc, train_diagnosis_acc, train_combined_acc = train_epoch(
        model, train_loader, optimizer, device, scaler
    )

    # Validation
    (val_loss, val_organ_acc, val_diagnosis_acc, val_combined_acc,
     val_organ_preds, val_organ_labels, val_diagnosis_preds, val_diagnosis_labels) = validate_epoch(
        model, val_loader, device
    )

    # Update scheduler
    scheduler.step()

    # Store history
    train_history['loss'].append(train_loss)
    train_history['organ_acc'].append(train_organ_acc)
    train_history['diagnosis_acc'].append(train_diagnosis_acc)
    train_history['combined_acc'].append(train_combined_acc)

    train_history['val_loss'].append(val_loss)
    train_history['val_organ_acc'].append(val_organ_acc)
    train_history['val_diagnosis_acc'].append(val_diagnosis_acc)
    train_history['val_combined_acc'].append(val_combined_acc)

    # Print epoch results
    print(f"Train - Loss: {train_loss:.4f}, Organ Acc: {train_organ_acc:.4f}, Diag Acc: {train_diagnosis_acc:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Organ Acc: {val_organ_acc:.4f}, Diag Acc: {val_diagnosis_acc:.4f}")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_organ_acc': val_organ_acc,
            'val_diagnosis_acc': val_diagnosis_acc,
        }, '/content/best_efficientnet_blip_model.pth')
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

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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
axes[1, 0].plot(train_history['diagnosis_acc'], label='Train Diagnosis Acc', marker='o', linewidth=2)
axes[1, 0].plot(train_history['val_diagnosis_acc'], label='Val Diagnosis Acc', marker='s', linewidth=2)
axes[1, 0].set_title('Diagnosis Classification Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Combined Accuracy
axes[1, 1].plot(train_history['combined_acc'], label='Train Combined Acc', marker='o', linewidth=2)
axes[1, 1].plot(train_history['val_combined_acc'], label='Val Combined Acc', marker='s', linewidth=2)
axes[1, 1].set_title('Combined Classification Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('EfficientNet-B3 + BLIP Training History', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/content/efficientnet_blip_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Training history saved as 'efficientnet_blip_training_history.png'")

# ==================== 11. TEST EVALUATION ====================
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

# Load best model
checkpoint = torch.load('/content/best_efficientnet_blip_model.pth')
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

# Truncate labels for display
organ_labels_display = [c[:12] for c in organ_classes_test]

sns.heatmap(organ_cm_normalized, annot=True, fmt='.2f', cmap='viridis',
            ax=axes[0], cbar_kws={'label': 'Normalized Count'},
            xticklabels=organ_labels_display,
            yticklabels=organ_labels_display)
axes[0].set_title('Organ Classification - EfficientNet + BLIP', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Organ', fontsize=10)
axes[0].set_ylabel('True Organ', fontsize=10)
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# Diagnosis Confusion Matrix
diagnosis_cm = confusion_matrix(test_diagnosis_labels, test_diagnosis_preds, labels=unique_diagnoses_test)
diagnosis_cm_normalized = diagnosis_cm.astype('float') / diagnosis_cm.sum(axis=1)[:, np.newaxis]
diagnosis_cm_normalized = np.nan_to_num(diagnosis_cm_normalized)

diagnosis_labels_display = [c[:12] for c in diagnosis_classes_test]

sns.heatmap(diagnosis_cm_normalized, annot=True, fmt='.2f', cmap='plasma',
            ax=axes[1], cbar_kws={'label': 'Normalized Count'},
            xticklabels=diagnosis_labels_display,
            yticklabels=diagnosis_labels_display)
axes[1].set_title('Diagnosis Classification - EfficientNet + BLIP', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Diagnosis', fontsize=10)
axes[1].set_ylabel('True Diagnosis', fontsize=10)
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

plt.suptitle('EfficientNet-B3 + BLIP Medical VQA - Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/content/efficientnet_blip_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Confusion matrices saved as 'efficientnet_blip_confusion_matrices.png'")

# ==================== 13. CLASSIFICATION REPORTS ====================
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORTS")
print("="*60)

print("\n📋 ORGAN CLASSIFICATION REPORT:")
organ_report = classification_report(
    test_organ_labels,
    test_organ_preds,
    labels=unique_organs_test,
    target_names=organ_classes_test,
    digits=3,
    zero_division=0
)
print(organ_report)

print("\n📋 DIAGNOSIS CLASSIFICATION REPORT:")
diagnosis_report = classification_report(
    test_diagnosis_labels,
    test_diagnosis_preds,
    labels=unique_diagnoses_test,
    target_names=diagnosis_classes_test,
    digits=3,
    zero_division=0
)
print(diagnosis_report)

# Save reports
with open('/content/efficientnet_blip_organ_report.txt', 'w') as f:
    f.write("EfficientNet-B3 + BLIP - ORGAN CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Overall Accuracy: {test_organ_acc:.4f}\n\n")
    f.write(organ_report)

with open('/content/efficientnet_blip_diagnosis_report.txt', 'w') as f:
    f.write("EfficientNet-B3 + BLIP - DIAGNOSIS CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Overall Accuracy: {test_diagnosis_acc:.4f}\n\n")
    f.write(diagnosis_report)

print("✓ Classification reports saved")

# ==================== 14. ERROR ANALYSIS ====================
print("\n🔍 ERROR ANALYSIS")

# Calculate per-class accuracy
print("\n📊 ORGAN CLASS ACCURACY (Top 5):")
organ_accuracies = []
for i, organ_idx in enumerate(unique_organs_test):
    organ_name = organ_classes_test[i]
    mask = (test_organ_labels == organ_idx)
    if mask.sum() > 0:
        acc = (np.array(test_organ_preds)[mask] == organ_idx).mean()
        count = mask.sum()
        organ_accuracies.append((organ_name, acc, count))

organ_accuracies.sort(key=lambda x: x[1], reverse=True)

for organ_name, acc, count in organ_accuracies[:5]:
    print(f"  {organ_name:<12}: {acc:.3f} ({count} samples)")

print("\n📊 DIAGNOSIS CLASS ACCURACY (Top 5):")
diagnosis_accuracies = []
for i, diag_idx in enumerate(unique_diagnoses_test):
    diag_name = diagnosis_classes_test[i]
    mask = (test_diagnosis_labels == diag_idx)
    if mask.sum() > 0:
        acc = (np.array(test_diagnosis_preds)[mask] == diag_idx).mean()
        count = mask.sum()
        diagnosis_accuracies.append((diag_name, acc, count))

diagnosis_accuracies.sort(key=lambda x: x[1], reverse=True)

for diag_name, acc, count in diagnosis_accuracies[:5]:
    print(f"  {diag_name:<12}: {acc:.3f} ({count} samples)")

# ==================== 15. SAVE FINAL MODEL ====================
print("\n" + "="*60)
print("SAVING FINAL MODEL AND RESULTS")
print("="*60)

# Save final model
final_model_path = '/content/efficientnet_blip_final_model.pth'
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'organ_encoder': organ_encoder,
    'diagnosis_encoder': diagnosis_encoder,
    'combined_encoder': combined_encoder,
    'processor': processor,
    'test_results': {
        'organ_accuracy': test_organ_acc,
        'diagnosis_accuracy': test_diagnosis_acc,
        'combined_accuracy': test_combined_acc,
        'test_loss': test_loss
    },
    'train_history': train_history
}, final_model_path)

print(f"✅ Final model saved as '{final_model_path}'")

# Save all artifacts
import pickle
with open('/content/efficientnet_blip_preprocessing.pkl', 'wb') as f:
    pickle.dump({
        'organ_encoder': organ_encoder,
        'diagnosis_encoder': diagnosis_encoder,
        'combined_encoder': combined_encoder,
        'organ_classes': organ_classes,
        'diagnosis_classes': diagnosis_classes,
        'combined_classes': combined_classes,
        'train_history': train_history
    }, f)
print("✅ Preprocessing artifacts saved")

# ==================== 16. MODEL INFERENCE EXAMPLE ====================
print("\n" + "="*60)
print("MODEL INFERENCE EXAMPLE")
print("="*60)

# Example inference
model.eval()
example_idx = 0
example_batch = next(iter(test_loader))

with torch.no_grad():
    images = example_batch['image'][:3].to(device)
    questions = example_batch['question'][:3]

    organ_logits, diagnosis_logits, _ = model(images, questions)

    organ_preds = organ_logits.argmax(dim=1).cpu().numpy()
    diagnosis_preds = diagnosis_logits.argmax(dim=1).cpu().numpy()

    print("\n📝 INFERENCE EXAMPLES:")
    for i in range(3):
        true_organ = organ_classes[example_batch['organ_label'][i].item()]
        true_diagnosis = diagnosis_classes[example_batch['diagnosis_label'][i].item()]
        pred_organ = organ_classes[organ_preds[i]]
        pred_diagnosis = diagnosis_classes[diagnosis_preds[i]]

        print(f"\nExample {i+1}:")
        print(f"  Question: {questions[i][:60]}...")
        print(f"  True: {true_organ} - {true_diagnosis}")
        print(f"  Pred: {pred_organ} - {pred_diagnosis}")
        print(f"  Correct: {true_organ == pred_organ and true_diagnosis == pred_diagnosis}")

# ==================== 17. FINAL SUMMARY ====================
print("\n" + "="*60)
print("PROJECT SUMMARY - EfficientNet + BLIP")
print("="*60)

print(f"\n📊 DATASET:")
print(f"  Total samples: {len(df)}")
print(f"  Training samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Organ classes: {len(organ_classes)}")
print(f"  Diagnosis classes: {len(diagnosis_classes)}")

print(f"\n🏗️  MODEL ARCHITECTURE:")
print(f"  Visual Encoder: EfficientNet-B3 (pretrained)")
print(f"  Vision-Language Model: BLIP (Salesforce/blip-vqa-base)")
print(f"  Feature Fusion: Projection + Concatenation")
print(f"  Multi-task Heads: Organ + Diagnosis + Combined")
print(f"  Total parameters: {total_params:,}")

print(f"\n🎯 FINAL TEST RESULTS:")
print(f"  Organ Classification Accuracy:    {test_organ_acc:.4f}")
print(f"  Diagnosis Classification Accuracy: {test_diagnosis_acc:.4f}")
print(f"  Combined Classification Accuracy:  {test_combined_acc:.4f}")

print(f"\n💾 SAVED FILES:")
files = [
    'efficientnet_blip_final_model.pth',
    'best_efficientnet_blip_model.pth',
    'efficientnet_blip_preprocessing.pkl',
    'efficientnet_blip_training_history.png',
    'efficientnet_blip_confusion_matrices.png',
    'efficientnet_blip_organ_report.txt',
    'efficientnet_blip_diagnosis_report.txt'
]

for file in files:
    print(f"  ✓ {file}")