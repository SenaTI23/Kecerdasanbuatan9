import nltk
from nltk.tokenize import word_tokenize
import spacy
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Download NLTK data
nltk.download('punkt')

# Load spaCy model Bahasa Indonesia
try:
    nlp = spacy.load("id_core_news_sm")
except:
    from spacy.cli import download
    download("id_core_news_sm")
    nlp = spacy.load("id_core_news_sm")

# Dataset 100 kalimat Bahasa Indonesia
corpus = [
    "Pelayanan restoran ini sangat memuaskan",
    "Makanan enak tapi harga mahal",
    "Produk elektronik ini mudah rusak",
    "Pengiriman cepat dan packaging rapi",
    "Saya kecewa dengan kualitas barang",
    "Toko online ini sangat profesional",
    "Barang tidak sesuai deskripsi",
    "Pelanggan puas dengan after-sales service",
    "Material terasa premium dan tahan lama",
    "Aplikasi mobile sering error saat checkout"
] * 10  # Duplikasi untuk contoh (praktiknya gunakan data asli)

# 1. Tokenisasi dengan NLTK
print("=== Hasil Tokenisasi NLTK ===")
nltk_tokens = [word_tokenize(text) for text in corpus[:5]]  # Ambil 5 sampel
for i, tokens in enumerate(nltk_tokens):
    print(f"Kalimat {i+1}: {tokens}")

# 2. Tokenisasi dengan spaCy
print("\n=== Hasil Tokenisasi spaCy ===")
spacy_tokens = [[token.text for token in nlp(text)] for text in corpus[:5]]
for i, tokens in enumerate(spacy_tokens):
    print(f"Kalimat {i+1}: {tokens}")

# 3. Pelatihan Word2Vec
sentences = [word_tokenize(text.lower()) for text in corpus]
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Contoh embedding kata "pelayanan"
print("\nEmbedding kata 'pelayanan':")
print(model_w2v.wv["pelayanan"])

# 4. Visualisasi PCA
words = ["pelayanan", "makanan", "produk", "pengiriman", "kualitas", "harga", "barang", "toko"]
vectors = np.array([model_w2v.wv[word] for word in words])

pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

plt.figure(figsize=(10, 6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title("Visualisasi Word Embeddings")
plt.show()