import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const metadata: Metadata = {
  title: 'Makine Öğrenmesi | Python Veri Bilimi | Kodleon',
  description: 'Python ile makine öğrenmesi. Denetimli ve denetimsiz öğrenme algoritmaları, model değerlendirme ve hiperparametre optimizasyonu.',
};

const content = `
# Makine Öğrenmesi Temelleri

## Giriş

Makine öğrenmesi, bilgisayarların verilerden öğrenmesini ve bu öğrendiklerini kullanarak tahminler yapmasını sağlayan bir yapay zeka alt dalıdır. Bu derste, makine öğrenmesinin temellerini adım adım öğreneceğiz.

## Temel Kavramlar

### 1. Veri ve Özellikler
- **Veri (Data)**: Makine öğrenmesi modellerinin öğrenmek için kullandığı bilgi
- **Özellikler (Features)**: Verideki her bir değişken
- **Hedef (Target)**: Tahmin etmeye çalıştığımız değer

### 2. Model Türleri
- **Denetimli Öğrenme**: Etiketli veri ile öğrenme
- **Denetimsiz Öğrenme**: Etiketsiz veri ile öğrenme
- **Pekiştirmeli Öğrenme**: Ödül-ceza sistemi ile öğrenme

## Denetimli Öğrenme

### 1. Veri Hazırlama
\`\`\`python
# Adım 1: Gerekli kütüphaneleri içe aktarma
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Adım 2: Örnek veri seti oluşturma
# Iris veri setini kullanalım
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  # özellikler
y = iris.target  # hedef değişken
\`\`\`

**🔍 Açıklama:**
- Öncelikle gerekli kütüphaneleri projemize dahil ediyoruz
- Scikit-learn'den hazır Iris veri setini yüklüyoruz
- X değişkeni özellikleri (çiçeğin ölçüleri), y değişkeni hedef sınıfı (çiçek türü) temsil eder

### 2. Veri Ön İşleme
\`\`\`python
# Adım 1: Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Adım 2: Veri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
\`\`\`

**🔍 Açıklama:**
- Veriyi %80 eğitim, %20 test olarak ayırıyoruz
- StandardScaler ile verileri normalize ediyoruz (ortalama=0, standart sapma=1)
- fit_transform() eğitim verisi için, transform() test verisi için kullanılır

**⚠️ Önemli Not:** Test verisini ölçeklendirirken sadece transform() kullanıyoruz, çünkü test verisi eğitim sürecinde bilinmeyen veriyi temsil eder.

### 3. Model Seçimi ve Eğitimi

#### 3.1 Lojistik Regresyon
\`\`\`python
# Adım 1: Model oluşturma
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(random_state=42)

# Adım 2: Model eğitimi
model_lr.fit(X_train_scaled, y_train)

# Adım 3: Tahmin
y_pred_lr = model_lr.predict(X_test_scaled)
\`\`\`

**🔍 Model Parametreleri:**
- random_state: Sonuçların tekrarlanabilirliği için
- solver: Optimizasyon algoritması ('lbfgs', 'newton-cg', 'sag', 'saga')
- max_iter: Maksimum iterasyon sayısı

#### 3.2 Rastgele Orman
\`\`\`python
# Adım 1: Model oluşturma
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Adım 2: Model eğitimi
model_rf.fit(X_train_scaled, y_train)

# Adım 3: Tahmin
y_pred_rf = model_rf.predict(X_test_scaled)
\`\`\`

**🔍 Model Parametreleri:**
- n_estimators: Ağaç sayısı
- max_depth: Maksimum ağaç derinliği
- min_samples_split: Düğüm bölmek için gereken minimum örnek sayısı
- min_samples_leaf: Yaprak düğümde olması gereken minimum örnek sayısı

### 4. Model Değerlendirme
\`\`\`python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Adım 1: Doğruluk skorları
print("Lojistik Regresyon Doğruluk:", 
      accuracy_score(y_test, y_pred_lr))
print("Rastgele Orman Doğruluk:", 
      accuracy_score(y_test, y_pred_rf))

# Adım 2: Detaylı metrikler
print("\\nLojistik Regresyon Raporu:")
print(classification_report(y_test, y_pred_lr))

print("\\nRastgele Orman Raporu:")
print(classification_report(y_test, y_pred_rf))
\`\`\`

**🔍 Metrikler ve Anlamları:**
- **Accuracy (Doğruluk)**: Doğru tahmin edilen örneklerin oranı
- **Precision (Kesinlik)**: Pozitif tahminlerin ne kadarının gerçekten pozitif olduğu
- **Recall (Duyarlılık)**: Gerçek pozitiflerin ne kadarının doğru tahmin edildiği
- **F1-Score**: Precision ve Recall'un harmonik ortalaması

### 5. Model İyileştirme

#### 5.1 Çapraz Doğrulama
\`\`\`python
from sklearn.model_selection import cross_val_score

# 5-katlı çapraz doğrulama
cv_scores_lr = cross_val_score(model_lr, X_train_scaled, y_train, cv=5)
cv_scores_rf = cross_val_score(model_rf, X_train_scaled, y_train, cv=5)

print("Lojistik Regresyon CV Skorları:", cv_scores_lr.mean())
print("Rastgele Orman CV Skorları:", cv_scores_rf.mean())
\`\`\`

**🔍 Açıklama:**
- Çapraz doğrulama, modelin genelleme yeteneğini ölçer
- Veri 5 parçaya bölünür ve her seferinde 4 parça eğitim, 1 parça test için kullanılır
- Final skor, 5 farklı testin ortalamasıdır

#### 5.2 Hiperparametre Optimizasyonu
\`\`\`python
from sklearn.model_selection import GridSearchCV

# Parametre ızgarası
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train_scaled, y_train)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor:", grid_search.best_score_)
\`\`\`

**🔍 Grid Search Stratejisi:**
1. Belirlenen parametre kombinasyonlarını dener
2. Her kombinasyon için çapraz doğrulama yapar
3. En iyi sonucu veren parametre setini seçer

## Pratik Uygulamalar

### 1. Ev Fiyat Tahmini
[Ev Fiyat Tahmini Örneği](/kod-ornekleri/ev-fiyat-tahmini)

### 2. Müşteri Segmentasyonu
[Müşteri Segmentasyonu Örneği](/kod-ornekleri/musteri-segmentasyonu)

### 3. Duygu Analizi
[Duygu Analizi Örneği](/kod-ornekleri/duygu-analizi)

## Önerilen Kaynaklar

1. 📚 Scikit-learn Dokümantasyonu
2. 📖 Python Machine Learning (Sebastian Raschka)
3. 🎓 Coursera - Machine Learning Specialization
4. 💻 Kaggle Competitions

## Alıştırmalar

1. Farklı bir veri seti ile sınıflandırma modeli oluşturun
2. Hiperparametre optimizasyonu yapın
3. Farklı metriklerle model performansını değerlendirin
4. Veri ön işleme adımlarını değiştirerek sonuçları karşılaştırın

## Sıkça Sorulan Sorular

1. **S: Hangi model türünü seçmeliyim?**
   C: Veri setinizin büyüklüğü, problem tipi ve hesaplama kaynaklarınıza göre değişir.

2. **S: Overfitting nasıl önlenir?**
   C: Cross-validation, regularization ve veri artırma teknikleri kullanılabilir.

3. **S: Ne zaman derin öğrenme kullanmalıyım?**
   C: Büyük veri setlerinde ve karmaşık örüntülerde derin öğrenme tercih edilir.
`;

const learningPath = [
  {
    title: '1. Makine Öğrenmesi Temelleri',
    description: 'Temel makine öğrenmesi kavramlarını ve algoritmaları öğrenin.',
    topics: [
      'Denetimli ve denetimsiz öğrenme',
      'Model değerlendirme metrikleri',
      'Özellik mühendisliği',
      'Çapraz doğrulama',
      'Hiperparametre optimizasyonu',
    ],
    icon: '🤖',
    href: '/topics/python/veri-bilimi/makine-ogrenmesi/temeller'
  },
  {
    title: '2. Sınıflandırma ve Regresyon',
    description: 'Denetimli öğrenme algoritmalarını ve uygulamalarını keşfedin.',
    topics: [
      'Lojistik regresyon',
      'Karar ağaçları',
      'Random Forest',
      'SVM',
      'Gradient Boosting',
    ],
    icon: '📊',
    href: '/topics/python/veri-bilimi/makine-ogrenmesi/siniflandirma'
  },
  {
    title: '3. Kümeleme ve Boyut İndirgeme',
    description: 'Denetimsiz öğrenme tekniklerini ve uygulamalarını öğrenin.',
    topics: [
      'K-means kümeleme',
      'Hiyerarşik kümeleme',
      'DBSCAN',
      'PCA',
      't-SNE',
    ],
    icon: '🎯',
    href: '/topics/python/veri-bilimi/makine-ogrenmesi/kumeleme'
  },
  {
    title: '4. Model Optimizasyonu',
    description: 'Model performansını artırmak için ileri düzey teknikleri keşfedin.',
    topics: [
      'Grid ve Random Search',
      'Ensemble yöntemler',
      'Pipeline oluşturma',
      'Model seçimi',
      'Özellik seçimi',
    ],
    icon: '⚡',
    href: '/topics/python/veri-bilimi/makine-ogrenmesi/optimizasyon'
  }
];

export default function MakineOgrenmesiPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <Button variant="ghost" className="mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />
          <Link href="/topics/python/veri-bilimi">Geri Dön</Link>
        </Button>
      </div>
      
      <Card className="p-6">
        <MarkdownContent content={content} />
      </Card>
    </div>
  );
} 