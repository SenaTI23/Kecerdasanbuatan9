from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model dan tokenizer
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased")

# Contoh teks (20 sampel)
texts = [
    "Pelayanan memuaskan, produk berkualitas tinggi",
    "Sangat kecewa dengan pengiriman yang lambat",
    "Harga terjangkau untuk kualitas segini",
    "Tidak sesuai ekspektasi, barang cacat",
    "Respon penjual cepat dan ramah",
    "Kemasan rusak saat diterima",
    "Produk original seperti di deskripsi",
    "Pengiriman lebih cepat dari estimasi",
    "Barang tidak bisa digunakan sama sekali",
    "Saya akan beli lagi di toko ini",
    "Material terasa murah dan mudah patah",
    "Pengalaman belanja terburuk tahun ini",
    "Cocok untuk hadiah, packing rapi",
    "Fitur tidak bekerja sebagaimana mestinya",
    "Warna berbeda dari foto di iklan",
    "Pelayanan setelah penjualan sangat baik",
    "Toko ini tidak profesional",
    "Sangat direkomendasikan untuk pemula",
    "Garansi tidak berlaku meski barang rusak",
    "Kualitas setara dengan harga premium"
]

# Tokenize teks
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Prediksi
with torch.no_grad():
    outputs = model(**encoded_inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Hasil
for i, text in enumerate(texts):
    sentiment = "positif" if predictions[i, 1] > predictions[i, 0] else "negatif"
    confidence = np.max(predictions[i].numpy())
    print(f"Text: {text}")
    print(f"Sentimen: {sentiment} (Confidence: {confidence:.4f})\n")