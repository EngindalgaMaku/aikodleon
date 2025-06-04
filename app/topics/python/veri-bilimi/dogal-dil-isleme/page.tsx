import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python ile Doğal Dil İşleme | Python Veri Bilimi | Kodleon',
  description: 'Python kullanarak doğal dil işleme tekniklerini, metin analizi ve dil modelleme uygulamalarını öğrenin.',
};

const content = `
# Python ile Doğal Dil İşleme

Doğal dil işleme (NLP), bilgisayarların insan dilini anlama, işleme ve üretme yeteneğini geliştiren bir yapay zeka alt dalıdır. Bu bölümde, Python ile NLP uygulamalarını öğreneceğiz.

## Metin Ön İşleme

### Temel Metin İşleme

\`\`\`python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# NLTK verilerini indirme
nltk.download(['punkt', 'stopwords', 'wordnet'])

# Örnek metin
metin = """
Python programlama dili ile doğal dil işleme çok eğlenceli! 
NLTK kütüphanesi, metin işleme için harika araçlar sunuyor. 
Metinleri analiz etmek için çeşitli yöntemler kullanabiliriz.
"""

# Cümlelere ayırma
cumleler = sent_tokenize(metin)
print("Cümleler:", cumleler)

# Kelimelere ayırma
kelimeler = word_tokenize(metin)
print("\\nKelimeler:", kelimeler)

# Durak kelimeleri yükleme
durak_kelimeler = set(stopwords.words('turkish'))

# Lemmatizer oluşturma
lemmatizer = WordNetLemmatizer()

# Metin temizleme fonksiyonu
def metin_temizle(metin):
    # Küçük harfe çevirme
    metin = metin.lower()
    
    # Kelimelere ayırma
    kelimeler = word_tokenize(metin)
    
    # Noktalama işaretlerini ve durak kelimeleri kaldırma
    kelimeler = [kelime for kelime in kelimeler 
                 if kelime not in string.punctuation
                 and kelime not in durak_kelimeler]
    
    # Lemmatization
    kelimeler = [lemmatizer.lemmatize(kelime) for kelime in kelimeler]
    
    return kelimeler

# Temizlenmiş metin
temiz_kelimeler = metin_temizle(metin)
print("\\nTemizlenmiş kelimeler:", temiz_kelimeler)
\`\`\`

### Kelime Torbası (Bag of Words)

\`\`\`python
from sklearn.feature_extraction.text import CountVectorizer

# Örnek metinler
metinler = [
    "Python ile programlama öğreniyorum",
    "Doğal dil işleme Python ile yapılıyor",
    "Makine öğrenmesi çok ilginç"
]

# CountVectorizer oluşturma
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(metinler)

# Özellik isimleri
print("Kelimeler:", vectorizer.get_feature_names_out())
print("\\nKelime torbası matrisi:\\n", X.toarray())
\`\`\`

## Metin Temsili

### TF-IDF

\`\`\`python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF vectorizer
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(metinler)

# TF-IDF matrisini görüntüleme
print("TF-IDF matrisi:\\n", X_tfidf.toarray())
print("\\nÖzellikler:", tfidf.get_feature_names_out())
\`\`\`

### Word2Vec

\`\`\`python
from gensim.models import Word2Vec

# Örnek cümleler
cumleler = [
    ["python", "programlama", "dili", "öğreniyorum"],
    ["doğal", "dil", "işleme", "python", "ile", "yapılıyor"],
    ["makine", "öğrenmesi", "çok", "ilginç"]
]

# Word2Vec modeli eğitimi
model = Word2Vec(sentences=cumleler, 
                vector_size=100, 
                window=5, 
                min_count=1)

# Kelime vektörlerini görüntüleme
print("'python' kelimesinin vektörü:\\n", 
      model.wv['python'])

# Benzer kelimeleri bulma
print("\\n'python' kelimesine benzer kelimeler:")
print(model.wv.most_similar('python'))
\`\`\`

## Metin Sınıflandırma

### Duygu Analizi

\`\`\`python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Örnek veri
yorumlar = [
    "Film çok güzeldi, kesinlikle tavsiye ederim",
    "Vakit kaybı, hiç beğenmedim",
    "Muhteşem bir film, tekrar izleyeceğim",
    "Çok sıkıcıydı, pişman oldum",
    # ... daha fazla örnek
]
etiketler = [1, 0, 1, 0]  # 1: Pozitif, 0: Negatif

# TF-IDF dönüşümü
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(yorumlar)

# Veri bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, etiketler, test_size=0.2, random_state=42
)

# Model eğitimi
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
print(classification_report(y_test, y_pred))
\`\`\`

## Konu Modelleme

### LDA (Latent Dirichlet Allocation)

\`\`\`python
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Örnek dokümanlar
dokumanlar = [
    "Python programlama dili çok kullanışlı",
    "Yapay zeka ve makine öğrenmesi popüler",
    "Doğal dil işleme metin analizi yapar",
    "Python ile veri analizi yapılabilir",
    "Makine öğrenmesi algoritmaları önemli"
]

# TF-IDF dönüşümü
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(dokumanlar)

# LDA modeli
n_topics = 2
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# Konuları görüntüleme
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] 
                 for i in topic.argsort()[:-10-1:-1]]
    print(f"Konu {topic_idx + 1}: {', '.join(top_words)}")
\`\`\`

## Dil Modelleri

### N-gram Modeli

\`\`\`python
from nltk import ngrams
from collections import Counter

# Örnek metin
metin = "Python ile doğal dil işleme öğreniyorum"
kelimeler = metin.split()

# Bigram oluşturma
bigramlar = list(ngrams(kelimeler, 2))
print("Bigramlar:", bigramlar)

# Trigram oluşturma
trigramlar = list(ngrams(kelimeler, 3))
print("\\nTrigramlar:", trigramlar)

# N-gram frekansları
bigram_freq = Counter(bigramlar)
print("\\nBigram frekansları:", dict(bigram_freq))
\`\`\`

### Transformers ile Modern Dil Modelleri

\`\`\`python
from transformers import pipeline

# Duygu analizi pipeline'ı
sentiment_analyzer = pipeline("sentiment-analysis")

# Metin üretme pipeline'ı
text_generator = pipeline("text-generation")

# Duygu analizi örneği
metin = "Bu ürün beklentilerimi fazlasıyla karşıladı!"
sonuc = sentiment_analyzer(metin)
print("Duygu analizi sonucu:", sonuc)

# Metin üretme örneği
prompt = "Python programlama"
uretilen_metin = text_generator(prompt, max_length=50)
print("\\nÜretilen metin:", uretilen_metin[0]['generated_text'])
\`\`\`

## Metin Özetleme

### Özetleyici Model

\`\`\`python
from transformers import pipeline

# Özetleme pipeline'ı
summarizer = pipeline("summarization")

# Örnek metin
uzun_metin = """
Python, veri bilimi ve yapay zeka alanında en popüler programlama dillerinden 
biridir. Zengin kütüphane ekosistemi ve kolay öğrenilebilir syntax'ı sayesinde 
hem başlangıç hem de ileri düzey projeler için tercih edilir. Özellikle 
TensorFlow ve PyTorch gibi derin öğrenme framework'leri, scikit-learn gibi 
makine öğrenmesi kütüphaneleri ve pandas gibi veri analizi araçları Python'ı 
veri bilimi için vazgeçilmez kılmaktadır.
"""

# Metin özetleme
ozet = summarizer(uzun_metin, max_length=75, min_length=30)
print("Özet:", ozet[0]['summary_text'])
\`\`\`

## Alıştırmalar

1. **Metin Ön İşleme**
   - Farklı dillerde metin temizleme yapın
   - Özel karakterleri ve emojileri işleyin
   - Kök bulma algoritmalarını karşılaştırın

2. **Metin Sınıflandırma**
   - Spam tespiti modeli geliştirin
   - Çok sınıflı metin sınıflandırma yapın
   - Farklı özellik çıkarma yöntemlerini deneyin

3. **Dil Modelleri**
   - Kendi N-gram modelinizi oluşturun
   - Transfer öğrenme ile fine-tuning yapın
   - Metin üretme modeli geliştirin

## Sonraki Adımlar

1. [Bilgisayarlı Görü](/topics/python/veri-bilimi/bilgisayarli-goru)
2. [Pekiştirmeli Öğrenme](/topics/python/veri-bilimi/pekistirmeli-ogrenme)
3. [Büyük Veri](/topics/python/veri-bilimi/buyuk-veri)

## Faydalı Kaynaklar

- [NLTK Dokümantasyonu](https://www.nltk.org/)
- [spaCy Öğreticileri](https://spacy.io/usage/spacy-101)
- [Hugging Face Transformers](https://huggingface.co/docs)
`;

export default function NLPPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/veri-bilimi" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Veri Bilimi
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert">
          <MarkdownContent content={content} />
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 