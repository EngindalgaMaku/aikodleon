import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Derin Öğrenme Deployment | Python Veri Bilimi | Kodleon',
  description: 'Derin öğrenme modellerinin optimizasyonu, dağıtımı ve izlenmesi için kapsamlı rehber.',
};

const content = `
# Derin Öğrenme Deployment

Bu bölümde, derin öğrenme modellerinin üretim ortamına nasıl deploy edileceğini ve optimize edileceğini öğreneceğiz.

## Model Optimizasyonu ve Dönüşümü

\`\`\`python
import torch
import torch.onnx
import onnx
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

class ModelOptimizasyonu:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def pytorch_to_onnx(self, model, ornek_girdi, dosya_adi, dinamik_axes=None):
        """PyTorch modelini ONNX formatına dönüştürür"""
        model.eval()
        
        # Dinamik boyutlar için varsayılan ayarlar
        if dinamik_axes is None:
            dinamik_axes = {
                'girdi': {0: 'batch_size'},
                'cikti': {0: 'batch_size'}
            }
        
        # ONNX dışa aktarma
        torch.onnx.export(
            model,
            ornek_girdi,
            dosya_adi,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['girdi'],
            output_names=['cikti'],
            dynamic_axes=dinamik_axes
        )
        
        # ONNX modelini doğrula
        onnx_model = onnx.load(dosya_adi)
        onnx.checker.check_model(onnx_model)
        print(f"Model başarıyla ONNX formatına dönüştürüldü ve kaydedildi: {dosya_adi}")
        
    def tensorflow_to_savedmodel(self, model, dosya_yolu):
        """TensorFlow modelini SavedModel formatına dönüştürür"""
        # Modeli kaydet
        tf.saved_model.save(model, dosya_yolu)
        
        # Concrete function oluştur
        full_model = tf.function(lambda x: model(x))
        concrete_func = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )
        
        # Frozen GraphDef oluştur
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        print(f"Frozen model boyutu: {len(frozen_func.graph.as_graph_def().node)} nodes")
        
        return frozen_func
        
    def tensorflow_to_tflite(self, model, dosya_adi, quantize=False):
        """TensorFlow modelini TFLite formatına dönüştürür"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            # Quantization uygula
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        tflite_model = converter.convert()
        
        # Modeli kaydet
        with open(dosya_adi, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Model başarıyla TFLite formatına dönüştürüldü ve kaydedildi: {dosya_adi}")
        
    def model_boyutu_analiz(self, dosya_yolu):
        """Model dosyasının boyutunu analiz eder"""
        import os
        boyut = os.path.getsize(dosya_yolu)
        
        # Boyutu okunaklı formata dönüştür
        birimler = ['B', 'KB', 'MB', 'GB']
        indeks = 0
        while boyut >= 1024 and indeks < len(birimler)-1:
            boyut /= 1024
            indeks += 1
            
        return f"{boyut:.2f} {birimler[indeks]}"

# Kullanım örneği
if __name__ == "__main__":
    optimizasyon = ModelOptimizasyonu()
    
    # PyTorch model örneği
    class BasitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
            
        def forward(self, x):
            return self.fc(x)
    
    model = BasitModel()
    ornek_girdi = torch.randn(1, 10)
    
    # ONNX dönüşümü
    optimizasyon.pytorch_to_onnx(model, ornek_girdi, "model.onnx")
    print("Model boyutu:", optimizasyon.model_boyutu_analiz("model.onnx"))
\`\`\`

## Model Serving ve API

\`\`\`python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import onnxruntime
import io
from PIL import Image
import json
from typing import List, Dict
import time
from prometheus_client import Counter, Histogram, start_http_server

class ModelServing:
    def __init__(self, model_yolu: str):
        # ONNX Runtime başlat
        self.session = onnxruntime.InferenceSession(
            model_yolu,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Metrikler
        self.tahmin_sayaci = Counter(
            'model_tahmin_sayisi',
            'Toplam tahmin sayısı'
        )
        self.tahmin_suresi = Histogram(
            'model_tahmin_suresi_saniye',
            'Tahmin süresi (saniye)'
        )
        
        # FastAPI uygulaması
        self.app = FastAPI(
            title="Model Serving API",
            description="Derin öğrenme modeli için REST API",
            version="1.0.0"
        )
        
        # CORS ayarları
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Route'ları ekle
        self.route_ekle()
        
    def route_ekle(self):
        @self.app.post("/tahmin")
        async def tahmin(dosya: UploadFile = File(...)):
            # Dosyayı oku ve ön işle
            icerik = await dosya.read()
            goruntu = Image.open(io.BytesIO(icerik))
            goruntu = self.goruntu_on_isle(goruntu)
            
            # Tahmin yap ve süreyi ölç
            with self.tahmin_suresi.time():
                tahminler = self.tahmin_yap(goruntu)
            
            # Metriği güncelle
            self.tahmin_sayaci.inc()
            
            return {
                "dosya_adi": dosya.filename,
                "tahminler": tahminler
            }
            
        @self.app.get("/saglik")
        def saglik_kontrolu():
            return {"durum": "çalışıyor"}
            
        @self.app.get("/metrikler")
        def metrikler():
            return {
                "tahmin_sayisi": self.tahmin_sayaci._value.get(),
                "ortalama_sure": self.tahmin_suresi._sum.get() / 
                                max(self.tahmin_suresi._count.get(), 1)
            }
    
    def goruntu_on_isle(self, goruntu: Image.Image) -> np.ndarray:
        # Görüntüyü yeniden boyutlandır
        goruntu = goruntu.resize((224, 224))
        # NumPy dizisine dönüştür ve normalize et
        goruntu = np.array(goruntu).astype('float32') / 255.0
        # Batch boyutu ekle
        goruntu = np.expand_dims(goruntu, axis=0)
        return goruntu
        
    def tahmin_yap(self, goruntu: np.ndarray) -> List[Dict[str, float]]:
        # ONNX Runtime ile tahmin
        girdi_adi = self.session.get_inputs()[0].name
        cikti_adi = self.session.get_outputs()[0].name
        
        tahminler = self.session.run(
            [cikti_adi],
            {girdi_adi: goruntu}
        )[0]
        
        # Tahminleri olasılıklara dönüştür
        olasiliklar = self._softmax(tahminler[0])
        
        # Sonuçları formatla
        return [
            {"sinif": i, "olasilik": float(p)}
            for i, p in enumerate(olasiliklar)
        ]
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    def basla(self, host: str = "0.0.0.0", port: int = 8000):
        # Prometheus metrik sunucusunu başlat
        start_http_server(9090)
        # FastAPI uygulamasını başlat
        uvicorn.run(self.app, host=host, port=port)

# Kullanım örneği
if __name__ == "__main__":
    model_servisi = ModelServing("model.onnx")
    model_servisi.basla()
\`\`\`

## Model İzleme ve Performans Analizi

\`\`\`python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import requests

class ModelIzleme:
    def __init__(self):
        self.metrikler = {
            'dogruluk': [],
            'hassasiyet': [],
            'duyarlilik': [],
            'f1': [],
            'tahmin_suresi': [],
            'bellek_kullanimi': [],
            'timestamp': []
        }
        
    def performans_olc(self, 
                      gercek_degerler: np.ndarray,
                      tahminler: np.ndarray) -> Dict[str, float]:
        """Model performans metriklerini hesaplar"""
        dogruluk = accuracy_score(gercek_degerler, tahminler)
        hassasiyet, duyarlilik, f1, _ = precision_recall_fscore_support(
            gercek_degerler,
            tahminler,
            average='weighted'
        )
        
        return {
            'dogruluk': dogruluk,
            'hassasiyet': hassasiyet,
            'duyarlilik': duyarlilik,
            'f1': f1
        }
        
    def metrik_kaydet(self, 
                     performans_metrikleri: Dict[str, float],
                     tahmin_suresi: float,
                     bellek_kullanimi: float):
        """Metrikleri kaydeder"""
        self.metrikler['dogruluk'].append(performans_metrikleri['dogruluk'])
        self.metrikler['hassasiyet'].append(performans_metrikleri['hassasiyet'])
        self.metrikler['duyarlilik'].append(performans_metrikleri['duyarlilik'])
        self.metrikler['f1'].append(performans_metrikleri['f1'])
        self.metrikler['tahmin_suresi'].append(tahmin_suresi)
        self.metrikler['bellek_kullanimi'].append(bellek_kullanimi)
        self.metrikler['timestamp'].append(datetime.now())
        
    def drift_tespit(self, 
                     yeni_veriler: np.ndarray,
                     referans_veriler: np.ndarray,
                     esik: float = 0.05) -> bool:
        """Veri kayması tespiti yapar"""
        from scipy.stats import ks_2samp
        
        # Kolmogorov-Smirnov testi
        _, p_value = ks_2samp(yeni_veriler.ravel(), referans_veriler.ravel())
        
        return p_value < esik
        
    def performans_raporu_olustur(self, 
                                baslangic_tarih: datetime = None,
                                bitis_tarih: datetime = None) -> Dict:
        """Belirli bir zaman aralığı için performans raporu oluşturur"""
        df = pd.DataFrame(self.metrikler)
        
        if baslangic_tarih:
            df = df[df['timestamp'] >= baslangic_tarih]
        if bitis_tarih:
            df = df[df['timestamp'] <= bitis_tarih]
            
        rapor = {
            'ortalama_metrikler': {
                'dogruluk': df['dogruluk'].mean(),
                'hassasiyet': df['hassasiyet'].mean(),
                'duyarlilik': df['duyarlilik'].mean(),
                'f1': df['f1'].mean(),
                'tahmin_suresi': df['tahmin_suresi'].mean(),
                'bellek_kullanimi': df['bellek_kullanimi'].mean()
            },
            'trend_analizi': {
                'dogruluk_trend': self._trend_hesapla(df['dogruluk']),
                'tahmin_suresi_trend': self._trend_hesapla(df['tahmin_suresi'])
            },
            'anomaliler': self._anomali_tespit(df)
        }
        
        return rapor
        
    def _trend_hesapla(self, seri: pd.Series) -> str:
        """Basit trend analizi yapar"""
        if len(seri) < 2:
            return "Yetersiz veri"
            
        ilk_yarim = seri[:len(seri)//2].mean()
        son_yarim = seri[len(seri)//2:].mean()
        
        fark = son_yarim - ilk_yarim
        if abs(fark) < 0.01:
            return "Stabil"
        return "Artış" if fark > 0 else "Azalış"
        
    def _anomali_tespit(self, df: pd.DataFrame) -> Dict[str, List[datetime]]:
        """Basit anomali tespiti yapar"""
        anomaliler = {}
        
        for metrik in ['dogruluk', 'tahmin_suresi', 'bellek_kullanimi']:
            seri = df[metrik]
            mean = seri.mean()
            std = seri.std()
            
            # 3 standart sapma dışındaki değerleri anomali kabul et
            anomali_indexler = seri[abs(seri - mean) > 3 * std].index
            anomaliler[metrik] = [
                df['timestamp'][i].strftime('%Y-%m-%d %H:%M:%S')
                for i in anomali_indexler
            ]
            
        return anomaliler
        
    def grafik_ciz(self, metrik: str):
        """Seçilen metrik için zaman serisi grafiği çizer"""
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.metrikler['timestamp'],
            self.metrikler[metrik],
            marker='o'
        )
        plt.title(f'{metrik.capitalize()} Zaman Serisi')
        plt.xlabel('Zaman')
        plt.ylabel(metrik.capitalize())
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Kullanım örneği
if __name__ == "__main__":
    izleyici = ModelIzleme()
    
    # Örnek veriler
    gercek = np.array([0, 1, 1, 0, 1])
    tahmin = np.array([0, 1, 1, 1, 1])
    
    # Performans ölçümü
    performans = izleyici.performans_olc(gercek, tahmin)
    
    # Metrik kaydı
    izleyici.metrik_kaydet(
        performans,
        tahmin_suresi=0.1,
        bellek_kullanimi=150.5
    )
    
    # Rapor oluştur
    rapor = izleyici.performans_raporu_olustur()
    print(json.dumps(rapor, indent=2))
\`\`\`

## Alıştırmalar

1. **Model Optimizasyonu**
   - Farklı quantization stratejilerini deneyin
   - Model pruning uygulayın
   - TensorRT optimizasyonlarını ekleyin

2. **Model Serving**
   - Batch prediction desteği ekleyin
   - Otomatik ölçeklendirme ekleyin
   - Önbellek mekanizması ekleyin

3. **Model İzleme**
   - Özel metrikler ekleyin
   - Alarm sistemi kurun
   - Otomatik raporlama ekleyin

## Sonraki Adımlar

1. [MLOps ve DevOps](/topics/python/veri-bilimi/mlops)
2. [Büyük Veri İşleme](/topics/python/veri-bilimi/buyuk-veri)
3. [Yapay Zeka Projeleri](/topics/python/veri-bilimi/yapay-zeka-projeleri)

## Faydalı Kaynaklar

- [ONNX Dokümantasyonu](https://onnx.ai/docs/)
- [TensorRT Dokümantasyonu](https://developer.nvidia.com/tensorrt)
- [FastAPI Dokümantasyonu](https://fastapi.tiangolo.com/)
- [Prometheus Dokümantasyonu](https://prometheus.io/docs/)
`;

export default function DeepLearningDeploymentPage() {
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