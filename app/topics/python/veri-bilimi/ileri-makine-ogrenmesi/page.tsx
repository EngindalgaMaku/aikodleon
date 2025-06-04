import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'İleri Seviye Makine Öğrenmesi | Python Veri Bilimi | Kodleon',
  description: 'İleri seviye makine öğrenmesi teknikleri, topluluk öğrenme yöntemleri ve model optimizasyonu konularını öğrenin.',
};

const content = `
# İleri Seviye Makine Öğrenmesi

Bu bölümde, makine öğrenmesinin daha ileri seviye konularını ve teknikleri inceleyeceğiz. Topluluk öğrenme yöntemleri, model optimizasyonu ve üretim ortamına geçiş konularına odaklanacağız.

## Topluluk Öğrenme Yöntemleri

### Stacking Sınıflandırıcı

\`\`\`python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

class StackingSiniflandirici(BaseEstimator, ClassifierMixin):
    def __init__(self, siniflandiricilar=None, meta_siniflandirici=None, n_folds=5):
        self.siniflandiricilar = siniflandiricilar or [
            RandomForestClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42)
        ]
        self.meta_siniflandirici = meta_siniflandirici or LogisticRegression()
        self.n_folds = n_folds
        self.trained_classifiers = []
        
    def fit(self, X, y):
        # Meta özellikleri oluştur
        meta_features = np.zeros((X.shape[0], len(self.siniflandiricilar)))
        
        # K-fold çapraz doğrulama
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Her sınıflandırıcı için
        for i, clf in enumerate(self.siniflandiricilar):
            # Her katlama için
            for train_idx, val_idx in kf.split(X):
                # Eğitim ve doğrulama verisini ayır
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                
                # Modeli eğit
                clf.fit(X_train_fold, y_train_fold)
                
                # Tahminleri meta özellikler olarak kaydet
                meta_features[val_idx, i] = clf.predict_proba(X_val_fold)[:, 1]
        
        # Tüm veri üzerinde sınıflandırıcıları eğit
        self.trained_classifiers = []
        for clf in self.siniflandiricilar:
            fitted_clf = clf.fit(X, y)
            self.trained_classifiers.append(fitted_clf)
        
        # Meta sınıflandırıcıyı eğit
        self.meta_siniflandirici.fit(meta_features, y)
        
        return self
    
    def predict_proba(self, X):
        # Meta özellikleri oluştur
        meta_features = np.zeros((X.shape[0], len(self.siniflandiricilar)))
        
        # Her sınıflandırıcı için tahminleri al
        for i, clf in enumerate(self.trained_classifiers):
            meta_features[:, i] = clf.predict_proba(X)[:, 1]
        
        # Meta sınıflandırıcı ile son tahmini yap
        return self.meta_siniflandirici.predict_proba(meta_features)
    
    def predict(self, X):
        return self.meta_siniflandirici.predict(
            self.predict_proba(X)
        )

# Kullanım örneği
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Örnek veri oluştur
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                         n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Stacking sınıflandırıcıyı oluştur ve eğit
stacking = StackingSiniflandirici()
stacking.fit(X_train, y_train)

# Tahmin yap ve performansı değerlendir
y_pred = stacking.predict(X_test)
print(f"Doğruluk: {accuracy_score(y_test, y_pred):.4f}")
\`\`\`

### Özellik Seçimi ve Boyut Azaltma

\`\`\`python
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class OzellikSecici:
    def __init__(self, n_features=10, n_components=5):
        self.n_features = n_features
        self.n_components = n_components
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif,
            k=n_features
        )
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
    def fit_transform(self, X, y=None):
        # Veriyi ölçeklendir
        X_scaled = self.scaler.fit_transform(X)
        
        # Özellik seçimi
        if y is not None:
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled
            
        # PCA uygula
        X_reduced = self.pca.fit_transform(X_selected)
        
        # Açıklanan varyans oranını yazdır
        explained_variance_ratio = self.pca.explained_variance_ratio_
        print("Açıklanan varyans oranları:")
        for i, ratio in enumerate(explained_variance_ratio):
            print(f"Bileşen {i+1}: {ratio:.4f}")
        
        return X_reduced
    
    def transform(self, X):
        # Veriyi ölçeklendir
        X_scaled = self.scaler.transform(X)
        
        # Özellik seçimi
        X_selected = self.feature_selector.transform(X_scaled)
        
        # PCA uygula
        X_reduced = self.pca.transform(X_selected)
        
        return X_reduced

# Kullanım örneği
from sklearn.datasets import load_breast_cancer

# Veri setini yükle
data = load_breast_cancer()
X, y = data.data, data.target

# Özellik seçici oluştur ve uygula
selector = OzellikSecici(n_features=15, n_components=5)
X_reduced = selector.fit_transform(X, y)
print(f"\\nOrijinal boyut: {X.shape}")
print(f"İndirgenmiş boyut: {X_reduced.shape}")
\`\`\`

## Model Optimizasyonu

### Hiperparametre Optimizasyonu

\`\`\`python
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

class HiperparametreOptimizasyonu:
    def __init__(self, base_model=None, param_distributions=None, n_iter=100):
        self.base_model = base_model or RandomForestClassifier()
        self.param_distributions = param_distributions or {
            'n_estimators': randint(50, 500),
            'max_depth': [None] + list(range(5, 31)),
            'min_samples_split': randint(2, 21),
            'min_samples_leaf': randint(1, 11),
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        self.n_iter = n_iter
        self.best_model = None
        self.search = None
        
    def optimize(self, X, y, cv=5, scoring='accuracy', n_jobs=-1):
        # Rastgele arama ile optimizasyon
        self.search = RandomizedSearchCV(
            estimator=self.base_model,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42,
            verbose=1
        )
        
        # Arama işlemini gerçekleştir
        self.search.fit(X, y)
        
        # En iyi modeli kaydet
        self.best_model = self.search.best_estimator_
        
        # Sonuçları yazdır
        print("\\nEn iyi parametreler:")
        for param, value in self.search.best_params_.items():
            print(f"{param}: {value}")
        print(f"\\nEn iyi çapraz doğrulama skoru: {self.search.best_score_:.4f}")
        
        return self.best_model
    
    def get_results_df(self):
        import pandas as pd
        results = pd.DataFrame(self.search.cv_results_)
        results = results.sort_values('rank_test_score')
        return results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]

# Kullanım örneği
from sklearn.datasets import make_classification

# Veri seti oluştur
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                         n_redundant=5, random_state=42)

# Optimizasyon nesnesini oluştur ve çalıştır
optimizer = HiperparametreOptimizasyonu()
best_model = optimizer.optimize(X, y)

# Sonuçları görüntüle
results_df = optimizer.get_results_df()
print("\\nEn iyi 5 model:")
print(results_df.head())
\`\`\`

## Model Dağıtımı ve Üretim

### Model Servisleştirme

\`\`\`python
import joblib
from flask import Flask, request, jsonify
import numpy as np

class ModelServis:
    def __init__(self, model_path=None):
        self.app = Flask(__name__)
        self.model = None
        
        # Model yükleme
        if model_path:
            self.load_model(model_path)
        
        # API endpoint'leri tanımla
        self.app.route('/predict', methods=['POST'])(self.predict)
        self.app.route('/batch_predict', methods=['POST'])(self.batch_predict)
        
    def load_model(self, model_path):
        """Eğitilmiş modeli yükle"""
        self.model = joblib.load(model_path)
        
    def save_model(self, model, model_path):
        """Modeli kaydet"""
        joblib.dump(model, model_path)
        self.model = model
        
    def predict(self):
        """Tekli tahmin için endpoint"""
        try:
            # JSON verisini al
            data = request.get_json()
            features = np.array(data['features']).reshape(1, -1)
            
            # Tahmin yap
            prediction = self.model.predict(features)
            probability = self.model.predict_proba(features)
            
            return jsonify({
                'status': 'success',
                'prediction': int(prediction[0]),
                'probability': probability[0].tolist()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
            
    def batch_predict(self):
        """Toplu tahmin için endpoint"""
        try:
            # JSON verisini al
            data = request.get_json()
            features = np.array(data['features'])
            
            # Tahminleri yap
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            
            return jsonify({
                'status': 'success',
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist()
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
    
    def run(self, host='0.0.0.0', port=5000):
        """Servisi başlat"""
        self.app.run(host=host, port=port)

# Kullanım örneği
if __name__ == '__main__':
    # Model eğitimi (örnek)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Örnek veri ve model
    X, y = make_classification(n_samples=1000, n_features=20)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Servisi oluştur
    servis = ModelServis()
    
    # Modeli kaydet ve yükle
    servis.save_model(model, 'model.joblib')
    
    # Servisi başlat
    servis.run()

# API Kullanım Örneği (requests ile)
"""
import requests

# Tekli tahmin
response = requests.post('http://localhost:5000/predict',
                       json={'features': [1.0] * 20})
print(response.json())

# Toplu tahmin
response = requests.post('http://localhost:5000/batch_predict',
                       json={'features': [[1.0] * 20] * 5})
print(response.json())
"""
\`\`\`

## Model İzleme ve Bakım

### Model Performans İzleme

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import json
import logging

class ModelIzleyici:
    def __init__(self, model_name, metrics_file='model_metrics.json'):
        self.model_name = model_name
        self.metrics_file = metrics_file
        self.metrics_history = self._load_metrics()
        
        # Logging ayarları
        logging.basicConfig(
            filename=f'{model_name}_monitoring.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def _load_metrics(self):
        """Metrik geçmişini yükle"""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
            
    def _save_metrics(self):
        """Metrik geçmişini kaydet"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
            
    def log_predictions(self, y_true, y_pred, timestamp=None):
        """Tahmin metriklerini kaydet"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if self.model_name not in self.metrics_history:
            self.metrics_history[self.model_name] = {}
            
        self.metrics_history[self.model_name][timestamp] = metrics
        self._save_metrics()
        
        # Log mesajı
        logging.info(
            f"Model performans metrikleri kaydedildi - "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )
        
    def get_performance_trend(self, metric='accuracy', last_n=10):
        """Metrik trendini analiz et"""
        if self.model_name not in self.metrics_history:
            return None
            
        metrics_df = pd.DataFrame.from_dict(
            self.metrics_history[self.model_name],
            orient='index'
        )
        
        metrics_df.index = pd.to_datetime(metrics_df.index)
        metrics_df = metrics_df.sort_index()
        
        if len(metrics_df) < 2:
            return None
            
        # Son N kayıt
        recent_metrics = metrics_df[metric].tail(last_n)
        
        # Trend analizi
        trend = np.polyfit(range(len(recent_metrics)), recent_metrics, 1)[0]
        
        return {
            'current_value': recent_metrics.iloc[-1],
            'trend': trend,
            'trend_direction': 'increasing' if trend > 0 else 'decreasing',
            'alert': trend < 0 and recent_metrics.iloc[-1] < recent_metrics.mean()
        }
        
    def generate_report(self):
        """Performans raporu oluştur"""
        if self.model_name not in self.metrics_history:
            return "Metrik geçmişi bulunamadı"
            
        metrics_df = pd.DataFrame.from_dict(
            self.metrics_history[self.model_name],
            orient='index'
        )
        
        report = f"Model Performans Raporu - {self.model_name}\\n"
        report += f"Rapor tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
        
        # Genel istatistikler
        report += "Genel İstatistikler:\\n"
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            current = metrics_df[metric].iloc[-1]
            avg = metrics_df[metric].mean()
            std = metrics_df[metric].std()
            report += f"{metric.capitalize()}:\\n"
            report += f"  Güncel: {current:.4f}\\n"
            report += f"  Ortalama: {avg:.4f}\\n"
            report += f"  Std: {std:.4f}\\n"
            
        # Trend analizi
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            trend_info = self.get_performance_trend(metric)
            if trend_info:
                report += f"\\n{metric.capitalize()} Trend:\\n"
                report += f"  Yön: {trend_info['trend_direction']}\\n"
                report += f"  Uyarı: {'Evet' if trend_info['alert'] else 'Hayır'}\\n"
                
        return report

# Kullanım örneği
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Veri ve model
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Model izleyici
izleyici = ModelIzleyici("RandomForest_v1")

# Performans izleme
y_pred = model.predict(X_test)
izleyici.log_predictions(y_test, y_pred)

# Rapor oluşturma
print(izleyici.generate_report())
\`\`\`

## Alıştırmalar

1. **Topluluk Öğrenme**
   - Farklı topluluk yöntemlerini karşılaştırın
   - Özel bir stacking modeli geliştirin
   - Voting sınıflandırıcı implementasyonu yapın

2. **Model Optimizasyonu**
   - Bayesian optimizasyon uygulayın
   - Özel bir kayıp fonksiyonu geliştirin
   - Cross-validation stratejileri deneyin

3. **Model Dağıtımı**
   - Docker container'ı oluşturun
   - Load balancing implementasyonu yapın
   - Model versiyonlama sistemi geliştirin

## Sonraki Adımlar

1. [Veri Bilimi Projeleri](/topics/python/veri-bilimi/projeler)
2. [Derin Öğrenme Projeleri](/topics/python/veri-bilimi/derin-ogrenme-projeleri)
3. [MLOps ve DevOps](/topics/python/veri-bilimi/mlops)

## Faydalı Kaynaklar

- [Scikit-learn Dokümantasyonu](https://scikit-learn.org/stable/)
- [MLflow Dokümantasyonu](https://www.mlflow.org/docs/latest/index.html)
- [Flask Dokümantasyonu](https://flask.palletsprojects.com/)
`;

export default function AdvancedMLPage() {
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