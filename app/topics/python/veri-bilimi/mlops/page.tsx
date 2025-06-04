import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'MLOps ve DevOps | Python Veri Bilimi | Kodleon',
  description: 'Makine öğrenmesi projelerinde MLOps ve DevOps uygulamaları, CI/CD süreçleri ve otomatik test sistemleri.',
};

const content = `
# MLOps ve DevOps

Bu bölümde, makine öğrenmesi projelerinde MLOps ve DevOps uygulamalarını öğreneceğiz.

## CI/CD Pipeline Oluşturma

\`\`\`python
import os
import yaml
from typing import Dict, List
import subprocess
import logging

class MLOpsPipeline:
    def __init__(self, proje_adi: str):
        self.proje_adi = proje_adi
        self.config = self._varsayilan_config()
        
        # Logging ayarları
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _varsayilan_config(self) -> Dict:
        """Varsayılan CI/CD yapılandırması"""
        return {
            'stages': ['test', 'train', 'evaluate', 'deploy'],
            'python_version': '3.9',
            'dependencies': [
                'pytest',
                'black',
                'pylint',
                'scikit-learn',
                'pandas',
                'numpy'
            ],
            'test_command': 'pytest tests/',
            'train_script': 'src/train.py',
            'evaluate_script': 'src/evaluate.py',
            'model_path': 'models/',
            'docker_image': f'ml-model-{self.proje_adi}:latest'
        }
        
    def github_actions_olustur(self):
        """GitHub Actions workflow dosyası oluşturur"""
        workflow = {
            'name': f'{self.proje_adi} CI/CD Pipeline',
            'on': {
                'push': {'branches': ['main']},
                'pull_request': {'branches': ['main']}
            },
            'jobs': {
                'build-and-test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout repository',
                            'uses': 'actions/checkout@v2'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v2',
                            'with': {
                                'python-version': self.config['python_version']
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt'
                        },
                        {
                            'name': 'Run tests',
                            'run': self.config['test_command']
                        },
                        {
                            'name': 'Train model',
                            'run': f'python {self.config["train_script"]}'
                        },
                        {
                            'name': 'Evaluate model',
                            'run': f'python {self.config["evaluate_script"]}'
                        },
                        {
                            'name': 'Build Docker image',
                            'run': f'docker build -t {self.config["docker_image"]} .'
                        }
                    ]
                }
            }
        }
        
        # Workflow dosyasını kaydet
        os.makedirs('.github/workflows', exist_ok=True)
        with open('.github/workflows/ci-cd.yml', 'w') as f:
            yaml.dump(workflow, f)
            
        self.logger.info("GitHub Actions workflow dosyası oluşturuldu")
        
    def dockerfile_olustur(self):
        """Dockerfile oluşturur"""
        dockerfile = f"""
FROM python:{self.config['python_version']}-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY {self.config['model_path']} {self.config['model_path']}
COPY src/ src/

EXPOSE 8000

CMD ["python", "src/serve.py"]
"""
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile)
            
        self.logger.info("Dockerfile oluşturuldu")
        
    def test_betigi_olustur(self):
        """Test betiği şablonu oluşturur"""
        test_code = """
import pytest
import numpy as np
from src.model import Model

def test_model_tahmin():
    model = Model()
    X = np.random.random((10, 5))
    tahminler = model.tahmin(X)
    assert tahminler.shape[0] == 10
    
def test_model_egitim():
    model = Model()
    X = np.random.random((100, 5))
    y = np.random.randint(0, 2, 100)
    model.egit(X, y)
    assert hasattr(model, 'egitildi')
"""
        os.makedirs('tests', exist_ok=True)
        with open('tests/test_model.py', 'w') as f:
            f.write(test_code)
            
        self.logger.info("Test betiği oluşturuldu")
        
    def requirements_olustur(self):
        """Requirements dosyası oluşturur"""
        with open('requirements.txt', 'w') as f:
            for dep in self.config['dependencies']:
                f.write(f"{dep}\\n")
                
        self.logger.info("Requirements dosyası oluşturuldu")
        
    def proje_yapisini_olustur(self):
        """Temel proje yapısını oluşturur"""
        dizinler = ['src', 'tests', 'models', 'data', 'notebooks']
        for dizin in dizinler:
            os.makedirs(dizin, exist_ok=True)
            
        self.logger.info("Proje yapısı oluşturuldu")
        
    def git_baslat(self):
        """Git repository'sini başlatır"""
        try:
            subprocess.run(['git', 'init'], check=True)
            with open('.gitignore', 'w') as f:
                f.write("""
__pycache__/
*.py[cod]
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
.env
.venv
venv/
ENV/
models/*.pkl
data/raw/
""")
            self.logger.info("Git repository başlatıldı")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git başlatma hatası: {str(e)}")

# Kullanım örneği
if __name__ == "__main__":
    pipeline = MLOpsPipeline("iris-siniflandirma")
    
    # Proje yapısını oluştur
    pipeline.proje_yapisini_olustur()
    
    # CI/CD dosyalarını oluştur
    pipeline.github_actions_olustur()
    pipeline.dockerfile_olustur()
    pipeline.test_betigi_olustur()
    pipeline.requirements_olustur()
    
    # Git repository'sini başlat
    pipeline.git_baslat()
\`\`\`

## Model Versiyonlama ve Takip

\`\`\`python
import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os

class ModelVersionlama:
    def __init__(self, deney_adi: str):
        self.deney_adi = deney_adi
        
        # MLflow ayarları
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(deney_adi)
        
    def deney_baslat(self, 
                    etiketler: Optional[Dict[str, str]] = None) -> None:
        """Yeni bir deney başlatır"""
        mlflow.start_run(tags=etiketler)
        
    def parametre_kaydet(self, parametreler: Dict[str, Any]) -> None:
        """Model parametrelerini kaydeder"""
        mlflow.log_params(parametreler)
        
    def metrik_kaydet(self, 
                     metrik_adi: str, 
                     deger: float, 
                     adim: Optional[int] = None) -> None:
        """Metrik değerini kaydeder"""
        mlflow.log_metric(metrik_adi, deger, step=adim)
        
    def model_kaydet(self, 
                    model: Any, 
                    model_adi: str,
                    metadata: Optional[Dict] = None) -> None:
        """Modeli kaydeder"""
        # Model meta verilerini kaydet
        if metadata:
            mlflow.log_dict(metadata, "metadata.json")
        
        # Modeli kaydet
        mlflow.sklearn.log_model(model, model_adi)
        
    def artifact_kaydet(self, 
                       dosya_yolu: str, 
                       artifact_yolu: Optional[str] = None) -> None:
        """Artifact'ı kaydeder"""
        mlflow.log_artifact(dosya_yolu, artifact_yolu)
        
    def deney_bitir(self) -> None:
        """Deneyi sonlandırır"""
        mlflow.end_run()
        
    def model_yukle(self, 
                   run_id: str, 
                   model_adi: str = "model") -> Any:
        """Kaydedilmiş modeli yükler"""
        return mlflow.sklearn.load_model(f"runs:/{run_id}/{model_adi}")
        
    def deney_karsilastir(self, 
                         metrik: str, 
                         n_deney: int = 5) -> pd.DataFrame:
        """Son n deneyi karşılaştırır"""
        client = mlflow.tracking.MlflowClient()
        deneyler = client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name(self.deney_adi).experiment_id],
            order_by=[f"metrics.{metrik} DESC"]
        )
        
        sonuclar = []
        for deney in deneyler[:n_deney]:
            sonuclar.append({
                'run_id': deney.info.run_id,
                'tarih': datetime.fromtimestamp(deney.info.start_time/1000.0),
                metrik: deney.data.metrics.get(metrik, None),
                'parametreler': deney.data.params
            })
            
        return pd.DataFrame(sonuclar)

# Kullanım örneği
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import pandas as pd
    
    # Veri yükle
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2
    )
    
    # Model versiyonlama başlat
    versiyonlama = ModelVersionlama("iris-siniflandirma")
    
    # Deney başlat
    versiyonlama.deney_baslat({"dataset": "iris", "model_type": "random_forest"})
    
    try:
        # Model eğitimi
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Parametreleri kaydet
        versiyonlama.parametre_kaydet(model.get_params())
        
        # Tahmin ve değerlendirme
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Metriği kaydet
        versiyonlama.metrik_kaydet("accuracy", accuracy)
        
        # Modeli kaydet
        metadata = {
            "feature_names": iris.feature_names,
            "target_names": iris.target_names,
            "description": "Iris sınıflandırma modeli"
        }
        versiyonlama.model_kaydet(model, "iris_model", metadata)
        
    finally:
        # Deneyi bitir
        versiyonlama.deney_bitir()
    
    # Deneyleri karşılaştır
    karsilastirma = versiyonlama.deney_karsilastir("accuracy")
    print("Deney Karşılaştırması:")
    print(karsilastirma)
\`\`\`

## Otomatik Test ve Kalite Kontrol

\`\`\`python
import pytest
from typing import Any, Dict, List
import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd
import json
import logging
from dataclasses import dataclass

@dataclass
class TestSonucu:
    basarili: bool
    mesaj: str
    detaylar: Dict[str, Any]

class ModelTestSistemi:
    def __init__(self, model: BaseEstimator):
        self.model = model
        self.logger = logging.getLogger(__name__)
        
    def girdi_kontrol(self, 
                     X: np.ndarray,
                     beklenen_boyut: int) -> TestSonucu:
        """Girdi verilerini kontrol eder"""
        try:
            # Boyut kontrolü
            if X.shape[1] != beklenen_boyut:
                return TestSonucu(
                    basarili=False,
                    mesaj=f"Girdi boyutu uyumsuz: {X.shape[1]} != {beklenen_boyut}",
                    detaylar={"shape": X.shape}
                )
            
            # Veri tipi kontrolü
            if not np.issubdtype(X.dtype, np.number):
                return TestSonucu(
                    basarili=False,
                    mesaj="Girdi sayısal değil",
                    detaylar={"dtype": str(X.dtype)}
                )
            
            # Eksik değer kontrolü
            if np.isnan(X).any():
                return TestSonucu(
                    basarili=False,
                    mesaj="Girdi eksik değerler içeriyor",
                    detaylar={"nan_count": np.isnan(X).sum()}
                )
                
            return TestSonucu(
                basarili=True,
                mesaj="Girdi kontrolleri başarılı",
                detaylar={"shape": X.shape, "dtype": str(X.dtype)}
            )
            
        except Exception as e:
            return TestSonucu(
                basarili=False,
                mesaj=f"Girdi kontrolü hatası: {str(e)}",
                detaylar={"error": str(e)}
            )
            
    def tahmin_kontrol(self, 
                      tahminler: np.ndarray,
                      beklenen_siniflar: List) -> TestSonucu:
        """Tahminleri kontrol eder"""
        try:
            # Geçerli sınıf kontrolü
            gecersiz_tahminler = [t for t in tahminler if t not in beklenen_siniflar]
            if gecersiz_tahminler:
                return TestSonucu(
                    basarili=False,
                    mesaj="Geçersiz sınıf tahminleri",
                    detaylar={"gecersiz_tahminler": gecersiz_tahminler}
                )
            
            # Tahmin dağılımı kontrolü
            sinif_dagilimi = {
                str(sinif): (tahminler == sinif).sum()
                for sinif in beklenen_siniflar
            }
            
            return TestSonucu(
                basarili=True,
                mesaj="Tahmin kontrolleri başarılı",
                detaylar={"sinif_dagilimi": sinif_dagilimi}
            )
            
        except Exception as e:
            return TestSonucu(
                basarili=False,
                mesaj=f"Tahmin kontrolü hatası: {str(e)}",
                detaylar={"error": str(e)}
            )
            
    def performans_kontrol(self,
                          metrikler: Dict[str, float],
                          esik_degerler: Dict[str, float]) -> TestSonucu:
        """Performans metriklerini kontrol eder"""
        try:
            basarisiz_metrikler = {
                metrik: deger
                for metrik, deger in metrikler.items()
                if deger < esik_degerler.get(metrik, 0)
            }
            
            if basarisiz_metrikler:
                return TestSonucu(
                    basarili=False,
                    mesaj="Performans metrikleri eşik değerlerin altında",
                    detaylar={"basarisiz_metrikler": basarisiz_metrikler}
                )
                
            return TestSonucu(
                basarili=True,
                mesaj="Performans kontrolleri başarılı",
                detaylar={"metrikler": metrikler}
            )
            
        except Exception as e:
            return TestSonucu(
                basarili=False,
                mesaj=f"Performans kontrolü hatası: {str(e)}",
                detaylar={"error": str(e)}
            )
            
    def model_kaydet(self, 
                    dosya_yolu: str,
                    metadata: Dict[str, Any]) -> TestSonucu:
        """Model ve meta verileri kaydeder"""
        try:
            # Meta verileri kaydet
            with open(f"{dosya_yolu}_metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            # Modeli kaydet
            import joblib
            joblib.dump(self.model, f"{dosya_yolu}_model.pkl")
            
            return TestSonucu(
                basarili=True,
                mesaj="Model başarıyla kaydedildi",
                detaylar={"dosya_yolu": dosya_yolu}
            )
            
        except Exception as e:
            return TestSonucu(
                basarili=False,
                mesaj=f"Model kaydetme hatası: {str(e)}",
                detaylar={"error": str(e)}
            )
            
    def test_raporu_olustur(self, 
                           sonuclar: List[TestSonucu]) -> Dict:
        """Test sonuçlarından rapor oluşturur"""
        return {
            "ozet": {
                "toplam_test": len(sonuclar),
                "basarili_test": sum(1 for s in sonuclar if s.basarili),
                "basarisiz_test": sum(1 for s in sonuclar if not s.basarili)
            },
            "detaylar": [
                {
                    "basarili": s.basarili,
                    "mesaj": s.mesaj,
                    "detaylar": s.detaylar
                }
                for s in sonuclar
            ]
        }

# Kullanım örneği
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    # Model ve test sistemi oluştur
    model = RandomForestClassifier()
    test_sistemi = ModelTestSistemi(model)
    
    # Örnek veri
    X = np.random.random((100, 4))
    y = np.random.randint(0, 3, 100)
    
    # Model eğitimi
    model.fit(X, y)
    tahminler = model.predict(X)
    
    # Testleri çalıştır
    test_sonuclari = []
    
    # Girdi kontrolü
    test_sonuclari.append(
        test_sistemi.girdi_kontrol(X, beklenen_boyut=4)
    )
    
    # Tahmin kontrolü
    test_sonuclari.append(
        test_sistemi.tahmin_kontrol(tahminler, [0, 1, 2])
    )
    
    # Performans kontrolü
    metrikler = {
        "accuracy": accuracy_score(y, tahminler),
        "precision": precision_score(y, tahminler, average='weighted'),
        "recall": recall_score(y, tahminler, average='weighted')
    }
    
    esik_degerler = {
        "accuracy": 0.8,
        "precision": 0.8,
        "recall": 0.8
    }
    
    test_sonuclari.append(
        test_sistemi.performans_kontrol(metrikler, esik_degerler)
    )
    
    # Test raporu
    rapor = test_sistemi.test_raporu_olustur(test_sonuclari)
    print(json.dumps(rapor, indent=2))
\`\`\`

## Alıştırmalar

1. **CI/CD Pipeline**
   - Jenkins pipeline oluşturun
   - Kubernetes deployment ekleyin
   - Otomatik rollback ekleyin

2. **Model Versiyonlama**
   - A/B test sistemi ekleyin
   - Model karşılaştırma araçları geliştirin
   - Otomatik model seçimi ekleyin

3. **Test Sistemi**
   - Veri kalite testleri ekleyin
   - Performans benchmark testleri ekleyin
   - Güvenlik testleri ekleyin

## Sonraki Adımlar

1. [Derin Öğrenme Deployment](/topics/python/veri-bilimi/derin-ogrenme-deployment)
2. [Büyük Veri İşleme](/topics/python/veri-bilimi/buyuk-veri)
3. [Yapay Zeka Projeleri](/topics/python/veri-bilimi/yapay-zeka-projeleri)

## Faydalı Kaynaklar

- [MLflow Dokümantasyonu](https://mlflow.org/docs/latest/index.html)
- [GitHub Actions Dokümantasyonu](https://docs.github.com/en/actions)
- [Docker Dokümantasyonu](https://docs.docker.com/)
- [pytest Dokümantasyonu](https://docs.pytest.org/)
`;

export default function MLOpsPage() {
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