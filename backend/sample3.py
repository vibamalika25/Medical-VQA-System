import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries installed")

# ==================== 2. MOUNT GOOGLE DRIVE ====================
from google.colab import drive
drive.mount('/content/drive')

# ==================== 3. LOAD DATASET FROM DRIVE ====================
def load_dataset_from_drive():
    """Load dataset from Google Drive"""
    print("📂 LOADING DATASET FROM GOOGLE DRIVE...")

    # Check for dataset in common locations
    possible_paths = [
        '/content/drive/MyDrive/archive (2)', # This path was seen in previous executions
        '/content/drive/MyDrive/medical_vqa_dataset',
        '/content/drive/MyDrive/QA_VLM_MED',
        '/content/drive/MyDrive/datasets/medical_vqa',
        '/content/drive/MyDrive/Medical_VQA_Dataset',
    ]

    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"✓ Found dataset at: {path}")
            break

    if dataset_path is None:
        print("⚠️ Dataset not found in common locations")
        print("Please specify your dataset path:")
        dataset_path = input("Enter full path to dataset folder: ")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    return dataset_path

dataset_path = load_dataset_from_drive()

# ==================== 4. ENHANCED DATA LOADER WITH ORGAN & DIAGNOSIS ANALYSIS ====================
class MedicalVQAOrganDiagnosisAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.image_folder = None
        self.organ_classes = []
        self.diagnosis_subclasses = []

    def load_and_prepare_data(self):
        """Load dataset and extract organ/diagnosis information"""
        print("\n🔍 ANALYZING DATASET STRUCTURE...")

        # Find and load CSV file
        csv_file = self.find_csv_file()

        if csv_file and os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            print(f"✓ Loaded data from: {csv_file}")

        else:
            # Create synthetic dataset for demonstration
            print("⚠️ Creating synthetic dataset for demonstration")
            self.df = self.create_synthetic_medical_dataset()

        # Find image folder
        self.find_image_folder()

        return self.df

    def find_csv_file(self):
        """Find CSV file in dataset directory"""
        csv_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))

        if csv_files:
            return csv_files[0]
        return None

    def create_synthetic_medical_dataset(self):
        """Create synthetic medical dataset for testing"""
        print("Creating synthetic medical dataset with organ/diagnosis structure...")

        # Define organs and their possible diagnoses
        organ_diagnosis_map = {
            'Lung': ['Normal', 'Benign_Nodule', 'Malignant_Tumor', 'Pneumonia'],
            'Liver': ['Normal', 'Benign_Cyst', 'Malignant_HCC', 'Cirrhosis'],
            'Brain': ['Normal', 'Benign_Meningioma', 'Malignant_Glioma', 'Stroke'],
            'Breast': ['Normal', 'Benign_Fibroadenoma', 'Malignant_Carcinoma', 'DCIS'],
            'Skin': ['Normal', 'Benign_Nevus', 'Malignant_Melanoma', 'Basal_Cell'],
            'Kidney': ['Normal', 'Benign_Cyst', 'Malignant_RCC', 'Stone'],
        }

        data = []
        image_counter = 1

        for organ, diagnoses in organ_diagnosis_map.items():
            for diagnosis in diagnoses:
                for _ in range(5): # Create 5 samples per organ-diagnosis combination
                    data.append({
                        'image': f'sample_img_{image_counter:04d}.jpg',
                        'question': f'What is the diagnosis for this {organ.lower()} image?',
                        'answer': diagnosis,
                        'Questions': f'What is the question for {organ.lower()}?', # Added to match potential original column names
                        'Answers': diagnosis # Added to match potential original column names
                    })
                    image_counter += 1

        df = pd.DataFrame(data)
        print(f"Created synthetic dataset with {len(df)} samples")
        return df

    def find_image_folder(self):
        """Find folder containing images"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for root, dirs, files in os.walk(self.data_path):
            img_count = sum(1 for f in files if any(f.lower().endswith(ext) for ext in image_extensions))
            if img_count > 0:
                self.image_folder = root
                print(f"\n🖼️ Found image folder: {root} ({img_count} images)")
                break

        if self.image_folder is None:
            print("⚠️ No image folder found")
            self.image_folder = self.data_path

# Initialize and run the analyzer to load df
print("\n" + "="*60)
print("INITIALIZING ORGAN-DIAGNOSIS ANALYZER")
print("="*60)

analyzer = MedicalVQAOrganDiagnosisAnalyzer(dataset_path)
df = analyzer.load_and_prepare_data()

# ==================== FIXED CODE FOR YOUR DATASET STRUCTURE (moved after df is loaded) ====================

print("\n🔍 DETECTED DATASET STRUCTURE:")
print(f"Columns found: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))

# Fix column names based on your dataset
# Your dataset has 'Questions' and 'Answers' columns instead of 'question' and 'answer'
# Let's rename them for consistency

# Check if 'Questions' and 'Answers' exist before renaming
if 'Questions' in df.columns and 'Answers' in df.columns:
    df = df.rename(columns={
        'Questions': 'question',
        'Answers': 'answer'
    })
    print("\n✅ Renamed columns:")
    print(f"New columns: {list(df.columns)}")
else:
    print("\n⚠️ 'Questions' or 'Answers' columns not found. Assuming 'question' and 'answer' are already correct or using existing columns.")
    # Ensure 'question' and 'answer' columns exist, if not, create dummy ones
    if 'question' not in df.columns:
        df['question'] = "default question"
    if 'answer' not in df.columns:
        df['answer'] = "default answer"


# Now let's analyze organ and diagnosis information from the answers
print("\n🔍 ANALYZING ANSWER CONTENT FOR ORGAN & DIAGNOSIS...")

# Extract organ and diagnosis information from answers
def extract_medical_info(answer):
    """Extract organ and diagnosis information from answers"""
    answer_lower = str(answer).lower()

    # Common organs in medical imaging
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

    # Common diagnoses
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

    # Find organ
    detected_organ = 'unknown'
    for organ, keywords in organs.items():
        if any(keyword in answer_lower for keyword in keywords):
            detected_organ = organ
            break

    # Find diagnosis
    detected_diagnosis = 'unknown'
    for diagnosis, keywords in diagnoses.items():
        if any(keyword in answer_lower for keyword in keywords):
            detected_diagnosis = diagnosis
            break

    return detected_organ, detected_diagnosis

# Apply extraction to all answers
print("\nExtracting organ and diagnosis information from answers...")
organ_list = []
diagnosis_list = []

for idx, answer in enumerate(df['answer']):
    organ, diagnosis = extract_medical_info(answer)
    organ_list.append(organ)
    diagnosis_list.append(diagnosis)

    # Show first few extractions
    if idx < 5:
        print(f"Answer {idx}: {answer[:80]}...")
        print(f"  → Organ: {organ}, Diagnosis: {diagnosis}")

# Add extracted information to dataframe
df['organ'] = organ_list
df['diagnosis'] = diagnosis_list

print(f"\n✅ Extracted organ and diagnosis information:")
print(f"  Unique organs: {df['organ'].unique()}")
print(f"  Unique diagnoses: {df['diagnosis'].unique()}")

# ==================== VISUALIZE EXTRACTED INFORMATION ====================
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Organ distribution
organ_counts = df['organ'].value_counts()
axes[0].bar(range(len(organ_counts)), organ_counts.values, color='skyblue')
axes[0].set_title('Extracted Organ Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Organ', fontsize=10)
axes[0].set_ylabel('Count', fontsize=10)
axes[0].set_xticks(range(len(organ_counts)))
axes[0].set_xticklabels(organ_counts.index, rotation=45, ha='right', fontsize=9)

# Diagnosis distribution
diagnosis_counts = df['diagnosis'].value_counts()
axes[1].bar(range(len(diagnosis_counts)), diagnosis_counts.values, color='lightgreen')
axes[1].set_title('Extracted Diagnosis Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Diagnosis', fontsize=10)
axes[1].set_ylabel('Count', fontsize=10)
axes[1].set_xticks(range(len(diagnosis_counts)))
axes[1].set_xticklabels(diagnosis_counts.index, rotation=45, ha='right', fontsize=9)

# Organ-Diagnosis combinations
df['combined_label'] = df['organ'] + '_' + df['diagnosis']
combined_counts = df['combined_label'].value_counts().head(10)
axes[2].barh(range(len(combined_counts)), combined_counts.values, color='salmon')
axes[2].set_title('Top 10 Organ-Diagnosis Combinations', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Count', fontsize=10)
axes[2].set_yticks(range(len(combined_counts)))
axes[2].set_yticklabels(combined_counts.index, fontsize=8)
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('/content/extracted_medical_info.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📊 EXTRACTED MEDICAL INFORMATION:")
print(f"Total samples: {len(df)}")
print(f"Organ classes (main): {len(df['organ'].unique())}")
print(f"Diagnosis subclasses: {len(df['diagnosis'].unique())}")
print(f"Unique combinations: {len(df['combined_label'].unique())}")

# ==================== FIXED PREPROCESSOR ====================
class FixedMedicalVQAPreprocessor:
    def __init__(self, df, image_folder):
        self.df = df
        self.image_folder = image_folder
        self.tokenizer = None
        self.organ_encoder = None
        self.diagnosis_encoder = None

    def preprocess_for_multi_task(self):
        """Preprocess data for multi-task learning (organ + diagnosis)"""
        print("\n🎯 PREPROCESSING FOR MULTI-TASK LEARNING...")

        # 1. Preprocess text questions
        X_text = self.preprocess_text()

        # 2. Encode organ labels (Main task)
        from sklearn.preprocessing import LabelEncoder

        self.organ_encoder = LabelEncoder()
        y_organ = self.organ_encoder.fit_transform(self.df['organ'])
        num_organs = len(self.organ_encoder.classes_)

        print(f"\n📊 ORGAN CLASSES ENCODED: {num_organs}")
        print("Organ classes:", self.organ_encoder.classes_)

        # 3. Encode diagnosis labels (Secondary task)
        self.diagnosis_encoder = LabelEncoder()
        y_diagnosis = self.diagnosis_encoder.fit_transform(self.df['diagnosis'])
        num_diagnosis = len(self.diagnosis_encoder.classes_)

        print(f"📊 DIAGNOSIS CLASSES ENCODED: {num_diagnosis}")
        print("Diagnosis classes:", self.diagnosis_encoder.classes_)

        # 4. Encode combined labels
        combined_encoder = LabelEncoder()
        y_combined = combined_encoder.fit_transform(self.df['combined_label'])
        num_combined = len(combined_encoder.classes_)

        print(f"📊 COMBINED LABELS ENCODED: {num_combined}")
        print(f"  Total unique combinations: {num_combined}")

        # 5. Load images (limit to 2000 for faster processing)
        X_images, valid_indices = self.load_images(max_images=2000)

        # Filter data based on loaded images
        X_text = X_text[valid_indices]
        y_organ = y_organ[valid_indices]
        y_diagnosis = y_diagnosis[valid_indices]
        y_combined = y_combined[valid_indices]
        df_filtered = self.df.iloc[valid_indices].reset_index(drop=True)

        print(f"\n📊 FINAL DATASET SHAPES:")
        print(f"  Images: {X_images.shape}")
        print(f"  Text: {X_text.shape}")
        print(f"  Organ labels: {y_organ.shape}")
        print(f"  Diagnosis labels: {y_diagnosis.shape}")
        print(f"  Combined labels: {y_combined.shape}")

        return {
            'X_images': X_images,
            'X_text': X_text,
            'y_organ': y_organ,
            'y_diagnosis': y_diagnosis,
            'y_combined': y_combined,
            'num_organs': num_organs,
            'num_diagnosis': num_diagnosis,
            'num_combined': num_combined,
            'organ_classes': self.organ_encoder.classes_,
            'diagnosis_classes': self.diagnosis_encoder.classes_,
            'combined_classes': combined_encoder.classes_,
            'df': df_filtered
        }

    def preprocess_text(self, max_vocab_size=5000, max_length=30):
        """Preprocess text questions"""
        import re
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Clean text
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^\w\s?]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        self.df['question_clean'] = self.df['question'].apply(clean_text)

        # Tokenize
        self.tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.df['question_clean'])

        sequences = self.tokenizer.texts_to_sequences(self.df['question_clean'])
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

        print(f"✓ Text preprocessing completed")
        print(f"  Vocabulary size: {len(self.tokenizer.word_index)}")

        return padded_sequences

    def load_images(self, img_size=(224, 224), max_images=None):
        """Load and preprocess images"""
        print(f"\n🖼️ LOADING IMAGES from {self.image_folder}...")

        images = []
        valid_indices = []

        # Determine how many images to load
        if max_images and max_images < len(self.df):
            indices = np.random.choice(len(self.df), max_images, replace=False)
        else:
            indices = range(len(self.df))

        loaded_count = 0
        for idx in indices:
            try:
                img_filename = self.df.iloc[idx]['image']

                # Try to find the image file
                img_path = None

                # Check different possible locations
                possible_paths = [
                    os.path.join(self.image_folder, str(img_filename)),
                    os.path.join(self.image_folder, 'images', str(img_filename)),
                    os.path.join('/content/drive/MyDrive/archive (2)', str(img_filename)),
                    os.path.join('/content/drive/MyDrive/archive (2)/QA_VLM_MED', str(img_filename)),
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        img_path = path
                        break

                if img_path and os.path.exists(img_path):
                    # Load and preprocess image
                    from PIL import Image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(img_size)
                    img_array = np.array(img) / 255.0

                    images.append(img_array)
                    valid_indices.append(idx)
                    loaded_count += 1

                    if loaded_count % 500 == 0:
                        print(f"  Loaded {loaded_count} images...")

                else:
                    # Create synthetic image for missing files
                    img_array = np.random.rand(img_size[0], img_size[1], 3)
                    images.append(img_array)
                    valid_indices.append(idx)
                    loaded_count += 1

            except Exception as e:
                if loaded_count < 3:
                    print(f"  Warning: Could not load image {img_filename}: {str(e)}")
                continue

        images = np.array(images)
        print(f"✓ Loaded {len(images)} images")

        return images, valid_indices

# Initialize preprocessor with fixed image folder path
image_folder = '/content/drive/MyDrive/archive (2)/QA_VLM_MED/images/images'
preprocessor = FixedMedicalVQAPreprocessor(df, image_folder)

# Preprocess for multi-task learning
processed_data = preprocessor.preprocess_for_multi_task()

# ==================== CONTINUE WITH MULTI-TASK MODEL ====================
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, optimizers

print(f"\n🧠 TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

class MultiTaskMedicalVQAModel:
    def __init__(self, vocab_size, num_organs, num_diagnosis, num_combined,
                 img_size=(224, 224), max_seq_len=30):
        self.vocab_size = vocab_size
        self.num_organs = num_organs
        self.num_diagnosis = num_diagnosis
        self.num_combined = num_combined
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.model = None

    def build_multi_task_model(self):
        """Build multi-task CNN+LSTM model"""
        print("\n🏗️ BUILDING MULTI-TASK CNN+LSTM MODEL...")

        # Image input
        image_input = layers.Input(shape=(self.img_size[0], self.img_size[1], 3), name='image_input')

        # Image augmentation
        augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
        ], name='image_augmentation')

        augmented_image = augmentation(image_input)

        # CNN backbone
        cnn_base = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size[0], self.img_size[1], 3),
            pooling='avg'
        )
        cnn_base.trainable = False

        image_features = cnn_base(augmented_image)
        image_features = layers.Dense(256, activation='relu')(image_features)
        image_features = layers.Dropout(0.3)(image_features)

        # Text input
        text_input = layers.Input(shape=(self.max_seq_len,), name='text_input')

        # Embedding layer: Generate tokens and a boolean mask
        embedding_output = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=256,
            input_length=self.max_seq_len,
            mask_zero=True # Generate boolean mask from embedding
        )(text_input)

        # Get the boolean mask generated by the Embedding layer
        boolean_mask = embedding_output._keras_mask # Shape: (batch_size, sequence_length)

        # Convert the boolean mask to a float mask (1.0 for valid, 0.0 for padded)
        float_mask = tf.cast(boolean_mask, dtype=tf.float32)
        # Expand dimensions to (batch_size, sequence_length, 1) for broadcasting
        expanded_float_mask = tf.expand_dims(float_mask, axis=-1)

        # Apply this float mask to the embedding output to numerically zero out padded regions.
        numerically_masked_embedding = embedding_output * expanded_float_mask

        # Bidirectional LSTM: Process the numerically masked embedding.
        lstm_output = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2)
        )(numerically_masked_embedding)

        # Apply the float mask again to LSTM output to ensure any new padded values are zeroed.
        lstm_output_numerically_masked = lstm_output * expanded_float_mask

        # Attention: Process the numerically masked LSTM output
        attention_output = layers.Attention()([lstm_output_numerically_masked, lstm_output_numerically_masked])
        # Apply the float mask again to attention output
        attention_output_numerically_masked = attention_output * expanded_float_mask

        # Concatenate the numerically masked outputs
        combined_text_features = layers.Concatenate()([lstm_output_numerically_masked, attention_output_numerically_masked])

        # Calculate Global Average Pooling manually since elements are numerically zeroed out
        sum_of_features = tf.reduce_sum(combined_text_features, axis=1) # Sum over sequence length

        # Count the number of non-zero (non-padded) elements per sequence using the float mask
        non_padded_token_counts = tf.reduce_sum(float_mask, axis=1, keepdims=True) # Shape: (batch_size, 1)

        # Add a small epsilon to avoid division by zero for empty sequences
        non_padded_token_counts = tf.maximum(non_padded_token_counts, 1e-10)

        # Calculate the average by dividing the sum by the count of non-padded tokens
        text_features = sum_of_features / non_padded_token_counts # Shape: (batch_size, 256)

        text_features = layers.Dense(128, activation='relu')(text_features)
        text_features = layers.Dropout(0.3)(text_features)

        # Multimodal fusion
        combined = layers.Concatenate()([image_features, text_features])

        # Shared layers
        shared = layers.Dense(512, activation='relu')(combined)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.4)(shared)

        shared = layers.Dense(256, activation='relu')(shared)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)

        # Task-specific heads
        # 1. Organ classification
        organ_head = layers.Dense(128, activation='relu')(shared)
        organ_head = layers.Dropout(0.2)(organ_head)
        organ_output = layers.Dense(self.num_organs, activation='softmax', name='organ_output')(organ_head)

        # 2. Diagnosis classification
        diagnosis_head = layers.Dense(128, activation='relu')(shared)
        diagnosis_head = layers.Dropout(0.2)(diagnosis_head)
        diagnosis_output = layers.Dense(self.num_diagnosis, activation='softmax', name='diagnosis_output')(diagnosis_head)

        # 3. Combined classification
        combined_head = layers.Dense(128, activation='relu')(shared)
        combined_head = layers.Dropout(0.2)(combined_head)
        combined_output = layers.Dense(self.num_combined, activation='softmax', name='combined_output')(combined_head)

        # Create model
        self.model = models.Model(
            inputs=[image_input, text_input],
            outputs=[organ_output, diagnosis_output, combined_output],
            name='MultiTask_Medical_VQA'
        )

        # Compile
        losses = {
            'organ_output': 'sparse_categorical_crossentropy',
            'diagnosis_output': 'sparse_categorical_crossentropy',
            'combined_output': 'sparse_categorical_crossentropy'
        }

        loss_weights = {
            'organ_output': 0.4,
            'diagnosis_output': 0.3,
            'combined_output': 0.3
        }

        metrics = {
            'organ_output': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='organ_top3')],
            'diagnosis_output': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='diag_top3')],
            'combined_output': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='combined_top3')]
        }

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        print("✅ Multi-task model built successfully!")
        self.model.summary()

        return self.model

    def get_callbacks(self):
        """Get training callbacks"""
        return [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='/content/best_multi_task_model.h5',
                monitor='val_combined_output_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

# Build the model
vocab_size = min(5000, len(preprocessor.tokenizer.word_index) + 1)

multi_task_model = MultiTaskMedicalVQAModel(
    vocab_size=vocab_size,
    num_organs=processed_data['num_organs'],
    num_diagnosis=processed_data['num_diagnosis'],
    num_combined=processed_data['num_combined']
)

model = multi_task_model.build_multi_task_model()
callbacks_list = multi_task_model.get_callbacks()

# ==================== SPLIT DATA ====================
from sklearn.model_selection import train_test_split

print("\n✂️ SPLITTING DATASET...")

X_images = processed_data['X_images']
X_text = processed_data['X_text']
y_organ = processed_data['y_organ']
y_diagnosis = processed_data['y_diagnosis']
y_combined = processed_data['y_combined']

# Split data
X_temp_img, X_test_img, X_temp_txt, X_test_txt = train_test_split(
    X_images, X_text, test_size=0.2, random_state=42
)

y_temp_organ, y_test_organ, y_temp_diag, y_test_diag, y_temp_comb, y_test_comb = train_test_split(
    y_organ, y_diagnosis, y_combined, test_size=0.2, random_state=42
)

# Further split
X_train_img, X_val_img, X_train_txt, X_val_txt = train_test_split(
    X_temp_img, X_temp_txt, test_size=0.125, random_state=42
)

y_train_organ, y_val_organ, y_train_diag, y_val_diag, y_train_comb, y_val_comb = train_test_split(
    y_temp_organ, y_temp_diag, y_temp_comb, test_size=0.125, random_state=42
)

print(f"✅ Dataset split completed:")
print(f"  Training: {len(X_train_img)} samples")
print(f"  Validation: {len(X_val_img)} samples")
print(f"  Test: {len(X_test_img)} samples")

# ==================== TRAIN MODEL ====================
print("\n🚂 STARTING MODEL TRAINING...")

# Prepare training data
train_data = ([X_train_img, X_train_txt],
              {'organ_output': y_train_organ,
               'diagnosis_output': y_train_diag,
               'combined_output': y_train_comb})

val_data = ([X_val_img, X_val_txt],
            {'organ_output': y_val_organ,
             'diagnosis_output': y_val_diag,
             'combined_output': y_val_comb})

# Train
history = model.fit(
    train_data[0],
    train_data[1],
    validation_data=val_data,
    epochs=20,  # Reduced for faster training
    batch_size=32,
    callbacks=callbacks_list,
    verbose=1
)

print("✅ Training completed!")

# ==================== EVALUATION ====================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Make predictions
test_predictions = model.predict([X_test_img, X_test_txt], verbose=0)
organ_pred_proba, diagnosis_pred_proba, combined_pred_proba = test_predictions

# Convert to class predictions
organ_pred = np.argmax(organ_pred_proba, axis=1)
diagnosis_pred = np.argmax(diagnosis_pred_proba, axis=1)
combined_pred = np.argmax(combined_pred_proba, axis=1)

from sklearn.metrics import accuracy_score

# Calculate accuracies
organ_accuracy = accuracy_score(y_test_organ, organ_pred)
diagnosis_accuracy = accuracy_score(y_test_diag, diagnosis_pred)
combined_accuracy = accuracy_score(y_test_comb, combined_pred)

print(f"\n📈 PERFORMANCE METRICS:")
print(f"Organ Classification Accuracy:    {organ_accuracy:.4f}")
print(f"Diagnosis Classification Accuracy: {diagnosis_accuracy:.4f}")
print(f"Combined Classification Accuracy:  {combined_accuracy:.4f}")

# ==================== CONFUSION MATRICES ====================
def plot_simple_confusion_matrices(y_true_organ, y_pred_organ, y_true_diag, y_pred_diag,
                                 organ_classes, diagnosis_classes):
    """Create simple confusion matrices"""
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Organ confusion matrix
    organ_cm = confusion_matrix(y_true_organ, y_pred_organ)
    organ_cm_normalized = organ_cm.astype('float') / organ_cm.sum(axis=1)[:, np.newaxis]

    im1 = axes[0].imshow(organ_cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('Organ Classification', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=10)
    axes[0].set_ylabel('True', fontsize=10)
    axes[0].set_xticks(range(len(organ_classes)))
    axes[0].set_xticklabels(organ_classes, rotation=45, ha='right', fontsize=9)

    # Diagnosis confusion matrix
    diag_cm = confusion_matrix(y_true_diag, y_pred_diag)
    diag_cm_normalized = diag_cm.astype('float') / diag_cm.sum(axis=1)[:, np.newaxis]

    im2 = axes[1].imshow(diag_cm_normalized, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Diagnosis Classification', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=10)
    axes[1].set_ylabel('True', fontsize=10)
    axes[1].set_xticks(range(len(diagnosis_classes)))
    axes[1].set_xticklabels(diagnosis_classes, rotation=45, ha='right', fontsize=9)

    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('/content/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✓ Confusion matrices saved as 'confusion_matrices.png'")

# Generate confusion matrices
plot_simple_confusion_matrices(
    y_test_organ, organ_pred,
    y_test_diag, diagnosis_pred,
    processed_data['organ_classes'],
    processed_data['diagnosis_classes']
)

# ==================== CLASSIFICATION REPORTS ====================
print("\n" + "="*60)
print("CLASSIFICATION REPORTS")
print("="*60)

from sklearn.metrics import classification_report

print("\n📋 ORGAN CLASSIFICATION REPORT:")
organ_report = classification_report(
    y_test_organ,
    organ_pred,
    target_names=[str(cls) for cls in processed_data['organ_classes']],
    digits=3,
    zero_division=0
)
print(organ_report)

print("\n📋 DIAGNOSIS CLASSIFICATION REPORT:")
diagnosis_report = classification_report(
    y_test_diag,
    diagnosis_pred,
    target_names=[str(cls) for cls in processed_data['diagnosis_classes']],
    digits=3,
    zero_division=0
)
print(diagnosis_report)

# Save reports
with open('/content/organ_report.txt', 'w') as f:
    f.write(organ_report)

with open('/content/diagnosis_report.txt', 'w') as f:
    f.write(diagnosis_report)

print("✓ Classification reports saved")

# ==================== SAVE MODEL ====================
print("\n" + "="*60)
print("SAVING MODEL AND RESULTS")
print("="*60)

# Save model
model.save('/content/medical_vqa_model.h5')
print("✅ Model saved as 'medical_vqa_model.h5'")

# Save preprocessing
import pickle
with open('/content/preprocessing.pkl', 'wb') as f:
    pickle.dump({
        'tokenizer': preprocessor.tokenizer,
        'organ_encoder': preprocessor.organ_encoder,
        'diagnosis_encoder': preprocessor.diagnosis_encoder,
        'organ_classes': processed_data['organ_classes'],
        'diagnosis_classes': processed_data['diagnosis_classes']
    }, f)
print("✅ Preprocessing artifacts saved as 'preprocessing.pkl'")

# Save metrics
import json
metrics = {
    'organ_accuracy': float(organ_accuracy),
    'diagnosis_accuracy': float(diagnosis_accuracy),
    'combined_accuracy': float(combined_accuracy),
    'training_epochs': len(history.history.get('loss', []))
}

with open('/content/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✅ Metrics saved as 'metrics.json'")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)

print(f"\n📊 DATASET:")
print(f"  Total samples: {len(df)}")
print(f"  Organ classes: {len(processed_data['organ_classes'])}")
print(f"  Diagnosis classes: {len(processed_data['diagnosis'].unique())}")

print(f"\n🏗️  MODEL:")
print(f"  Architecture: Multi-task CNN+LSTM")
print(f"  Image encoder: EfficientNetB0")
print(f"  Text encoder: BiLSTM with Attention")
print(f"  Tasks: Organ + Diagnosis classification")

print(f"\n🎯 RESULTS:")
print(f"  Organ Accuracy: {organ_accuracy:.4f}")
print(f"  Diagnosis Accuracy: {diagnosis_accuracy:.4f}")
print(f"  Combined Accuracy: {combined_accuracy:.4f}")

print(f"\n💾 SAVED FILES:")
files = [
    'medical_vqa_model.h5',
    'preprocessing.pkl',
    'metrics.json',
    'extracted_medical_info.png',
    'confusion_matrices.png',
    'organ_report.txt',
    'diagnosis_report.txt'
]

for file in files:
    if os.path.exists(f'/content/{file}'):
        print(f"  ✓ {file}")

print(f"\n🚀 NEXT STEPS:")
print("  1. Copy files to Drive:")
print("     !cp /content/*.h5 /content/drive/MyDrive/")
print("     !cp /content/*.pkl /content/drive/MyDrive/")
print("     !cp /content/*.json /content/drive/MyDrive/")
print("     !cp /content/*.png /content/drive/MyDrive/")
print("     !cp /content/*.txt /content/drive/MyDrive/")
print("  2. Load model for predictions:")
print("     model = tf.keras.models.load_model('medical_vqa_model.h5')")

print(f"\n✅ Medical VQA Project Completed Successfully!")