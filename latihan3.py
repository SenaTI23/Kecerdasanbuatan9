import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Dataset 500 Sampel ---
def generate_dataset():
    # Template kalimat positif dan negatif
    positif = [
        "Produk berkualitas tinggi", "Pengiriman sangat cepat", "Pelayanan memuaskan",
        "Harga terjangkau", "Barang sesuai deskripsi", "Packaging sangat aman",
        "Penjual ramah dan responsif", "Material premium", "Dikirim dengan bonus",
        "Pengalaman belanja menyenangkan"
    ]
    
    negatif = [
        "Kualitas di bawah standar", "Pengiriman terlambat", "Pelayanan buruk",
        "Harga terlalu mahal", "Barang tidak sesuai", "Packaging rusak",
        "Penjual tidak kooperatif", "Material mudah rusak", "Barang cacat",
        "Pengalaman belanja mengecewakan"
    ]
    
    # Generate 250 sampel positif dengan variasi
    data_positif = [
        f"{p} {random.choice(['', 'sangat', 'benar-benar', ''])} {random.choice(['recommended', 'puas', 'bagus'])}"
        for p in positif for _ in range(25)
    ][:250]  # Pastikan 250 sampel
    
    # Generate 250 sampel negatif dengan variasi
    data_negatif = [
        f"{n} {random.choice(['', 'sangat', 'benar-benar', ''])} {random.choice(['tidak direkomendasikan', 'kecewa', 'jelek'])}"
        for n in negatif for _ in range(25)
    ][:250]  # Pastikan 250 sampel
    
    df = pd.DataFrame({
        'text': data_positif + data_negatif,
        'label': [1]*250 + [0]*250
    })
    return df.sample(frac=1).reset_index(drop=True)  # Acak dataset

df = generate_dataset()
print(f"Jumlah total data: {len(df)}")
print(df['label'].value_counts())

# --- Bagi Dataset ---
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# --- Tokenizer ---
tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

train_encodings = tokenizer(
    train_texts.tolist(), 
    truncation=True, 
    padding=True, 
    max_length=128
)
val_encodings = tokenizer(
    val_texts.tolist(), 
    truncation=True, 
    padding=True, 
    max_length=128
)

# --- Dataset Class ---
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, train_labels.tolist())
val_dataset = ReviewDataset(val_encodings, val_labels.tolist())

# --- DataLoader ---
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# --- Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(
    'indolem/indobert-base-uncased', 
    num_labels=2
).to(device)

# --- Training ---
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Evaluasi
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    print(f"\nEpoch {epoch + 1}")
    print(classification_report(val_true, val_preds, target_names=['negatif', 'positif']))

# --- Simpan Model ---
model.save_pretrained("./model_sentimen")
tokenizer.save_pretrained("./model_sentimen")