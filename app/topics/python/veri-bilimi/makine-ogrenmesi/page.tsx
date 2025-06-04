import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Makine Öğrenmesi | Python Veri Bilimi | Kodleon',
  description: 'Python ile makine öğrenmesi. Denetimli ve denetimsiz öğrenme algoritmaları, model değerlendirme ve hiperparametre optimizasyonu.',
};

const content = `
# Makine Öğrenmesi

Bu bölümde, Python ile makine öğrenmesi uygulamalarını ve temel algoritmaları öğreneceğiz.

## Denetimli Öğrenme (Supervised Learning)

\`\`\`python
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
from dataclasses import dataclass
import logging

@dataclass
class ModelMetrikleri:
    accuracy: float
    precision: float
    recall: float
    f1: float
    
@dataclass
class ModelKonfigurasyonu:
    model_adi: str
    model_tipi: str
    parametreler: Dict[str, Any]
    egitim_parametreleri: Dict[str, Any]
    
class DenetimliOgrenme:
    def __init__(self, 
                 konfigurasyon: ModelKonfigurasyonu,
                 olceklendirme: bool = True):
        self.konfigurasyon = konfigurasyon
        self.olceklendirme = olceklendirme
        self.model = None
        self.olceklendirici = StandardScaler() if olceklendirme else None
        
        # Logging ayarları
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def veri_hazirla(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     test_orani: float = 0.2,
                     rastgele_durum: int = 42) -> Tuple[np.ndarray, ...]:
        """Veriyi eğitim ve test setlerine ayırır"""
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_orani, random_state=rastgele_durum
        )
        
        # Ölçeklendirme uygula
        if self.olceklendirme:
            X_train = self.olceklendirici.fit_transform(X_train)
            X_test = self.olceklendirici.transform(X_test)
            
        return X_train, X_test, y_train, y_test
        
    def model_olustur(self) -> BaseEstimator:
        """Konfigürasyona göre model oluşturur"""
        if self.konfigurasyon.model_tipi == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**self.konfigurasyon.parametreler)
        elif self.konfigurasyon.model_tipi == 'svm':
            from sklearn.svm import SVC
            return SVC(**self.konfigurasyon.parametreler)
        elif self.konfigurasyon.model_tipi == 'logistic':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**self.konfigurasyon.parametreler)
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {self.konfigurasyon.model_tipi}")
            
    def model_egit(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray) -> None:
        """Modeli eğitir"""
        self.model = self.model_olustur()
        self.model.fit(X_train, y_train)
        self.logger.info(f"{self.konfigurasyon.model_adi} eğitimi tamamlandı")
        
    def model_degerlendir(self,
                         X_test: np.ndarray,
                         y_test: np.ndarray) -> ModelMetrikleri:
        """Model performansını değerlendirir"""
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş")
            
        y_pred = self.model.predict(X_test)
        
        metrikler = ModelMetrikleri(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average='weighted'),
            recall=recall_score(y_test, y_pred, average='weighted'),
            f1=f1_score(y_test, y_pred, average='weighted')
        )
        
        self.logger.info(f"Model metrikleri hesaplandı: {metrikler}")
        return metrikler
        
    def capraz_dogrulama(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         k_fold: int = 5) -> Dict[str, List[float]]:
        """K-fold çapraz doğrulama uygular"""
        if self.model is None:
            self.model = self.model_olustur()
            
        metrikler = {
            'accuracy': cross_val_score(
                self.model, X, y, cv=k_fold, scoring='accuracy'
            ),
            'precision': cross_val_score(
                self.model, X, y, cv=k_fold, scoring='precision_weighted'
            ),
            'recall': cross_val_score(
                self.model, X, y, cv=k_fold, scoring='recall_weighted'
            ),
            'f1': cross_val_score(
                self.model, X, y, cv=k_fold, scoring='f1_weighted'
            )
        }
        
        for metrik, skorlar in metrikler.items():
            self.logger.info(
                f"{metrik}: {skorlar.mean():.4f} (+/- {skorlar.std() * 2:.4f})"
            )
            
        return metrikler
        
    def model_kaydet(self, dosya_yolu: str) -> None:
        """Modeli ve konfigürasyonu kaydeder"""
        if self.model is None:
            raise ValueError("Kaydedilecek model bulunamadı")
            
        model_verisi = {
            'model': self.model,
            'konfigurasyon': self.konfigurasyon,
            'olceklendirici': self.olceklendirici
        }
        
        joblib.dump(model_verisi, dosya_yolu)
        self.logger.info(f"Model kaydedildi: {dosya_yolu}")
        
    @classmethod
    def model_yukle(cls, dosya_yolu: str) -> 'DenetimliOgrenme':
        """Kaydedilmiş modeli yükler"""
        model_verisi = joblib.load(dosya_yolu)
        
        sinif = cls(
            konfigurasyon=model_verisi['konfigurasyon'],
            olceklendirme=model_verisi['olceklendirici'] is not None
        )
        
        sinif.model = model_verisi['model']
        sinif.olceklendirici = model_verisi['olceklendirici']
        
        return sinif

# Kullanım örneği
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    
    # Veri setini yükle
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Model konfigürasyonu
    konfigurasyon = ModelKonfigurasyonu(
        model_adi="Iris Sınıflandırıcı",
        model_tipi="random_forest",
        parametreler={
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42
        },
        egitim_parametreleri={
            'test_orani': 0.2,
            'k_fold': 5
        }
    )
    
    # Model oluştur ve eğit
    model = DenetimliOgrenme(konfigurasyon)
    
    # Veriyi hazırla
    X_train, X_test, y_train, y_test = model.veri_hazirla(
        X, y, 
        test_orani=konfigurasyon.egitim_parametreleri['test_orani']
    )
    
    # Modeli eğit
    model.model_egit(X_train, y_train)
    
    # Model performansını değerlendir
    metrikler = model.model_degerlendir(X_test, y_test)
    print("Model Metrikleri:", metrikler)
    
    # Çapraz doğrulama
    cv_metrikler = model.capraz_dogrulama(
        X, y, 
        k_fold=konfigurasyon.egitim_parametreleri['k_fold']
    )
    
    # Modeli kaydet
    model.model_kaydet("iris_model.joblib")
\`\`\`

## Denetimsiz Öğrenme (Unsupervised Learning)

\`\`\`python
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

@dataclass
class KumelemeMetrikleri:
    silhouette: float
    calinski_harabasz: float
    inertia: Optional[float] = None
    
@dataclass
class KumelemeKonfigurasyonu:
    model_adi: str
    model_tipi: str
    parametreler: Dict[str, Any]
    
class DenetimsizOgrenme:
    def __init__(self,
                 konfigurasyon: KumelemeKonfigurasyonu,
                 olceklendirme: bool = True):
        self.konfigurasyon = konfigurasyon
        self.olceklendirme = olceklendirme
        self.model = None
        self.olceklendirici = StandardScaler() if olceklendirme else None
        
        # Logging ayarları
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def veri_hazirla(self, X: np.ndarray) -> np.ndarray:
        """Veriyi hazırlar ve ölçeklendirir"""
        if self.olceklendirme:
            return self.olceklendirici.fit_transform(X)
        return X
        
    def model_olustur(self) -> BaseEstimator:
        """Konfigürasyona göre model oluşturur"""
        if self.konfigurasyon.model_tipi == 'kmeans':
            from sklearn.cluster import KMeans
            return KMeans(**self.konfigurasyon.parametreler)
        elif self.konfigurasyon.model_tipi == 'dbscan':
            from sklearn.cluster import DBSCAN
            return DBSCAN(**self.konfigurasyon.parametreler)
        elif self.konfigurasyon.model_tipi == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            return AgglomerativeClustering(**self.konfigurasyon.parametreler)
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {self.konfigurasyon.model_tipi}")
            
    def kumeleme_yap(self, X: np.ndarray) -> np.ndarray:
        """Kümeleme analizi yapar"""
        self.model = self.model_olustur()
        etiketler = self.model.fit_predict(X)
        self.logger.info(f"{self.konfigurasyon.model_adi} kümeleme tamamlandı")
        return etiketler
        
    def metrikleri_hesapla(self,
                          X: np.ndarray,
                          etiketler: np.ndarray) -> KumelemeMetrikleri:
        """Kümeleme metriklerini hesaplar"""
        metrikler = KumelemeMetrikleri(
            silhouette=silhouette_score(X, etiketler),
            calinski_harabasz=calinski_harabasz_score(X, etiketler)
        )
        
        # K-means için inertia hesapla
        if hasattr(self.model, 'inertia_'):
            metrikler.inertia = self.model.inertia_
            
        self.logger.info(f"Kümeleme metrikleri hesaplandı: {metrikler}")
        return metrikler
        
    def optimal_kume_sayisi(self,
                           X: np.ndarray,
                           min_k: int = 2,
                           max_k: int = 10) -> Dict[str, List[float]]:
        """Optimal küme sayısını bulmak için metrikler hesaplar"""
        if self.konfigurasyon.model_tipi != 'kmeans':
            raise ValueError("Bu metod sadece K-means için kullanılabilir")
            
        sonuclar = {
            'k': list(range(min_k, max_k + 1)),
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': []
        }
        
        for k in sonuclar['k']:
            self.konfigurasyon.parametreler['n_clusters'] = k
            etiketler = self.kumeleme_yap(X)
            metrikler = self.metrikleri_hesapla(X, etiketler)
            
            sonuclar['inertia'].append(metrikler.inertia)
            sonuclar['silhouette'].append(metrikler.silhouette)
            sonuclar['calinski_harabasz'].append(metrikler.calinski_harabasz)
            
        return sonuclar
        
    def kumeleri_gorselleştir(self,
                             X: np.ndarray,
                             etiketler: np.ndarray,
                             boyutlar: Tuple[int, int] = (0, 1)) -> None:
        """Kümeleri 2D düzlemde görselleştirir"""
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            X[:, boyutlar[0]], 
            X[:, boyutlar[1]], 
            c=etiketler, 
            cmap='viridis'
        )
        plt.colorbar(scatter)
        plt.title(f"{self.konfigurasyon.model_adi} Kümeleme Sonuçları")
        plt.xlabel(f"Özellik {boyutlar[0]}")
        plt.ylabel(f"Özellik {boyutlar[1]}")
        plt.show()

# Kullanım örneği
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    # Yapay veri oluştur
    X, _ = make_blobs(
        n_samples=300,
        n_features=2,
        centers=4,
        cluster_std=0.60,
        random_state=42
    )
    
    # Model konfigürasyonu
    konfigurasyon = KumelemeKonfigurasyonu(
        model_adi="Blob Kümeleme",
        model_tipi="kmeans",
        parametreler={
            'n_clusters': 4,
            'random_state': 42
        }
    )
    
    # Model oluştur
    model = DenetimsizOgrenme(konfigurasyon)
    
    # Veriyi hazırla
    X_hazir = model.veri_hazirla(X)
    
    # Kümeleme yap
    etiketler = model.kumeleme_yap(X_hazir)
    
    # Metrikleri hesapla
    metrikler = model.metrikleri_hesapla(X_hazir, etiketler)
    print("Kümeleme Metrikleri:", metrikler)
    
    # Optimal küme sayısını bul
    optimizasyon = model.optimal_kume_sayisi(X_hazir)
    
    # Sonuçları görselleştir
    model.kumeleri_gorselleştir(X_hazir, etiketler)
\`\`\`

## Alıştırmalar

1. **Denetimli Öğrenme**
   - Farklı veri setleri üzerinde sınıflandırma modelleri oluşturun
   - Hiperparametre optimizasyonu yapın
   - Model performansını değerlendirin ve karşılaştırın

2. **Denetimsiz Öğrenme**
   - Farklı kümeleme algoritmaları deneyin
   - Optimal küme sayısını belirleyin
   - Kümeleme sonuçlarını görselleştirin

3. **Model Geliştirme**
   - Özellik mühendisliği teknikleri uygulayın
   - Farklı ön işleme stratejileri deneyin
   - Model performansını artırmak için ensemble yöntemler kullanın

## Sonraki Adımlar

1. [Derin Öğrenme](/topics/python/veri-bilimi/derin-ogrenme)
2. [MLOps ve DevOps](/topics/python/veri-bilimi/mlops)
3. [Yapay Zeka Projeleri](/topics/python/veri-bilimi/yapay-zeka-projeleri)

## Faydalı Kaynaklar

- [scikit-learn Dokümantasyonu](https://scikit-learn.org/)
- [Python Machine Learning (Kitap)](https://www.packtpub.com/product/python-machine-learning-third-edition/9781789955750)
- [Kaggle Eğitimleri](https://www.kaggle.com/learn)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
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

        <h2 className="text-2xl font-bold mb-6">Öğrenme Yolu</h2>
        
        <div className="grid gap-6 md:grid-cols-2">
          {learningPath.map((topic, index) => (
            <Card key={index} className="p-6 hover:bg-accent transition-colors cursor-pointer">
              <Link href={topic.href}>
                <div className="flex items-start space-x-4">
                  <div className="text-4xl">{topic.icon}</div>
                  <div className="space-y-2">
                    <h3 className="font-bold">{topic.title}</h3>
                    <p className="text-sm text-muted-foreground">{topic.description}</p>
                    <ul className="text-sm space-y-1 list-disc list-inside text-muted-foreground">
                      {topic.topics.map((t, i) => (
                        <li key={i}>{t}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Link>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 