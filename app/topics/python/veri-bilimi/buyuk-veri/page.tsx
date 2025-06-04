import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Büyük Veri İşleme | Python Veri Bilimi | Kodleon',
  description: 'Dağıtık hesaplama, akış işleme ve veri boru hatları ile büyük veri işleme teknikleri.',
};

const content = `
# Büyük Veri İşleme

Bu bölümde, büyük veri işleme tekniklerini ve dağıtık hesaplama sistemlerini öğreneceğiz.

## PySpark ile Dağıtık Veri İşleme

\`\`\`python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, avg, count
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

class SparkVeriAnalizi:
    def __init__(self, app_name="BuyukVeriAnalizi"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
            
    def veri_yukle(self, dosya_yolu, format="csv"):
        """Veriyi Spark DataFrame'e yükler"""
        if format == "csv":
            return self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(dosya_yolu)
        elif format == "parquet":
            return self.spark.read.parquet(dosya_yolu)
        elif format == "json":
            return self.spark.read.json(dosya_yolu)
            
    def veri_on_isle(self, df):
        """Veri temizleme ve ön işleme"""
        # Eksik değerleri doldur
        df_temiz = df.na.fill(0)
        
        # Kategorik değişkenleri sayısallaştır
        kategorik_kolonlar = [f.name for f in df.schema.fields 
                            if isinstance(f.dataType, StringType)]
        
        for kolon in kategorik_kolonlar:
            df_temiz = df_temiz.withColumn(
                f"{kolon}_index",
                df_temiz[kolon].cast(IntegerType())
            )
            
        return df_temiz
        
    def kume_analizi(self, df, feature_kolonlari, k=3):
        """K-means kümeleme analizi yapar"""
        # Özellikleri birleştir
        assembler = VectorAssembler(
            inputCols=feature_kolonlari,
            outputCol="features"
        )
        df_vector = assembler.transform(df)
        
        # K-means modeli
        kmeans = KMeans(k=k, seed=1)
        model = kmeans.fit(df_vector)
        
        # Tahminler
        tahminler = model.transform(df_vector)
        
        return tahminler, model
        
    def sql_sorgula(self, df, sorgu):
        """SQL sorgusu çalıştırır"""
        df.createOrReplaceTempView("veri")
        return self.spark.sql(sorgu)
        
    def performans_analizi(self, df):
        """DataFrame performans metrikleri"""
        # Partition sayısı
        partition_sayisi = df.rdd.getNumPartitions()
        
        # Satır sayısı
        satir_sayisi = df.count()
        
        # Bellek kullanımı (yaklaşık)
        bellek = df.rdd.map(lambda x: len(str(x))).sum()
        
        return {
            "partition_sayisi": partition_sayisi,
            "satir_sayisi": satir_sayisi,
            "yaklasik_bellek_mb": bellek / (1024 * 1024)
        }
        
    def kapat(self):
        """Spark oturumunu kapatır"""
        self.spark.stop()

# Kullanım örneği
if __name__ == "__main__":
    analiz = SparkVeriAnalizi()
    
    # Veri yükle
    df = analiz.veri_yukle("buyuk_veri.csv")
    
    # Ön işleme
    df_temiz = analiz.veri_on_isle(df)
    
    # Kümeleme analizi
    feature_kolonlari = ["yas", "gelir", "harcama"]
    tahminler, model = analiz.kume_analizi(df_temiz, feature_kolonlari)
    
    # SQL sorgusu
    sonuc = analiz.sql_sorgula(
        tahminler,
        "SELECT prediction, COUNT(*) as sayi FROM veri GROUP BY prediction"
    )
    sonuc.show()
    
    # Performans analizi
    metrikler = analiz.performans_analizi(df)
    print("Performans Metrikleri:", metrikler)
    
    analiz.kapat()
\`\`\`

## Dask ile Paralel İşleme

\`\`\`python
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
import numpy as np
from typing import List, Dict, Any
import time

class DaskVeriIsleme:
    def __init__(self, n_workers=4):
        # Yerel küme oluştur
        self.cluster = LocalCluster(n_workers=n_workers)
        self.client = Client(self.cluster)
        
    def buyuk_veri_olustur(self, satir_sayisi: int, kolon_sayisi: int) -> dd.DataFrame:
        """Büyük veri seti oluşturur"""
        # Rastgele veri matrisi
        data = da.random.random((satir_sayisi, kolon_sayisi))
        
        # DataFrame'e dönüştür
        df = dd.from_array(
            data,
            columns=[f'ozellik_{i}' for i in range(kolon_sayisi)]
        )
        
        return df
        
    def paralel_hesapla(self, df: dd.DataFrame) -> Dict[str, Any]:
        """Paralel hesaplamalar yapar"""
        # Temel istatistikler
        sonuclar = {
            'ortalama': df.mean().compute(),
            'std': df.std().compute(),
            'min': df.min().compute(),
            'max': df.max().compute()
        }
        
        # Korelasyon matrisi
        sonuclar['korelasyon'] = df.corr().compute()
        
        return sonuclar
        
    def matris_carpimi(self, boyut: int) -> float:
        """Büyük matris çarpımı yapar"""
        # Rastgele matrisler
        A = da.random.random((boyut, boyut))
        B = da.random.random((boyut, boyut))
        
        # Çarpım zamanını ölç
        baslangic = time.time()
        sonuc = da.matmul(A, B)
        sonuc.compute()
        sure = time.time() - baslangic
        
        return sure
        
    def grup_analizi(self, df: dd.DataFrame, grup_kolonu: str) -> dd.DataFrame:
        """Grup bazlı analizler yapar"""
        return df.groupby(grup_kolonu).agg({
            'ozellik_0': ['mean', 'std', 'count']
        }).compute()
        
    def performans_karsilastir(self, islem_func, pandas_func, *args):
        """Dask ve Pandas performansını karşılaştırır"""
        # Dask ile ölç
        dask_baslangic = time.time()
        dask_sonuc = islem_func(*args)
        dask_sure = time.time() - dask_baslangic
        
        # Pandas ile ölç
        pandas_baslangic = time.time()
        pandas_sonuc = pandas_func(*args)
        pandas_sure = time.time() - pandas_baslangic
        
        return {
            'dask_suresi': dask_sure,
            'pandas_suresi': pandas_sure,
            'hizlanma': pandas_sure / dask_sure
        }
        
    def kapat(self):
        """Dask client'ı kapatır"""
        self.client.close()
        self.cluster.close()

# Kullanım örneği
if __name__ == "__main__":
    islemci = DaskVeriIsleme()
    
    # Büyük veri seti oluştur
    df = islemci.buyuk_veri_olustur(1000000, 10)
    
    # Paralel hesaplamalar
    sonuclar = islemci.paralel_hesapla(df)
    print("İstatistikler:", sonuclar)
    
    # Matris çarpımı
    sure = islemci.matris_carpimi(1000)
    print(f"Matris çarpımı süresi: {sure:.2f} saniye")
    
    islemci.kapat()
\`\`\`

## Kafka ile Veri Akışı İşleme

\`\`\`python
from kafka import KafkaProducer, KafkaConsumer
import json
from typing import Dict, List
import threading
import queue
import time
from datetime import datetime

class KafkaVeriAkisi:
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
        self.mesaj_kuyrugu = queue.Queue()
        self.calisma_durumu = False
        
    def producer_baslat(self):
        """Kafka producer'ı başlatır"""
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
    def consumer_baslat(self, topic: str, grup_id: str):
        """Kafka consumer'ı başlatır"""
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=grup_id,
            auto_offset_reset='latest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
    def mesaj_gonder(self, topic: str, mesaj: Dict):
        """Mesaj gönderir"""
        try:
            # Zaman damgası ekle
            mesaj['timestamp'] = datetime.now().isoformat()
            
            # Mesajı gönder
            future = self.producer.send(topic, mesaj)
            self.producer.flush()
            
            # Sonucu bekle
            record_metadata = future.get(timeout=10)
            
            return {
                'topic': record_metadata.topic,
                'partition': record_metadata.partition,
                'offset': record_metadata.offset
            }
        except Exception as e:
            print(f"Mesaj gönderme hatası: {str(e)}")
            return None
            
    def mesajlari_isle(self, topic: str, isleyici_fonksiyon):
        """Mesajları işler"""
        self.calisma_durumu = True
        
        def mesaj_dinleyici():
            for mesaj in self.consumer:
                if not self.calisma_durumu:
                    break
                    
                try:
                    # Mesajı işle
                    sonuc = isleyici_fonksiyon(mesaj.value)
                    # Sonucu kuyruğa ekle
                    self.mesaj_kuyrugu.put({
                        'veri': mesaj.value,
                        'sonuc': sonuc,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Mesaj işleme hatası: {str(e)}")
                    
        # İşleyici thread'i başlat
        thread = threading.Thread(target=mesaj_dinleyici)
        thread.start()
        
        return thread
        
    def mesaj_kuyrugunu_kontrol_et(self) -> List[Dict]:
        """İşlenmiş mesajları kuyruktan alır"""
        sonuclar = []
        
        while not self.mesaj_kuyrugu.empty():
            sonuclar.append(self.mesaj_kuyrugu.get())
            
        return sonuclar
        
    def durdur(self):
        """Akış işlemeyi durdurur"""
        self.calisma_durumu = False
        
        if self.producer:
            self.producer.close()
            
        if self.consumer:
            self.consumer.close()

# Kullanım örneği
if __name__ == "__main__":
    # Kafka işleyici oluştur
    akis = KafkaVeriAkisi()
    
    # Producer başlat
    akis.producer_baslat()
    
    # Consumer başlat
    akis.consumer_baslat('test-topic', 'test-group')
    
    # Örnek işleyici fonksiyon
    def mesaj_isleyici(mesaj):
        return {
            'uzunluk': len(str(mesaj)),
            'islenme_zamani': datetime.now().isoformat()
        }
    
    # İşlemeyi başlat
    thread = akis.mesajlari_isle('test-topic', mesaj_isleyici)
    
    # Örnek mesajlar gönder
    for i in range(5):
        mesaj = {'id': i, 'veri': f'Test mesajı {i}'}
        akis.mesaj_gonder('test-topic', mesaj)
        time.sleep(1)
        
        # İşlenmiş mesajları kontrol et
        sonuclar = akis.mesaj_kuyrugunu_kontrol_et()
        print(f"İşlenmiş mesajlar: {sonuclar}")
    
    # Sistemi durdur
    akis.durdur()
    thread.join()
\`\`\`

## Alıştırmalar

1. **PySpark**
   - Farklı veri formatlarıyla çalışın
   - SQL sorgularını optimize edin
   - Özel dönüşüm fonksiyonları yazın

2. **Dask**
   - Özel dağıtım stratejileri geliştirin
   - Bellek optimizasyonu yapın
   - Farklı kümeleme algoritmaları ekleyin

3. **Kafka**
   - Çoklu topic desteği ekleyin
   - Hata toleransı mekanizmaları ekleyin
   - Gerçek zamanlı analitik ekleyin

## Sonraki Adımlar

1. [MLOps ve DevOps](/topics/python/veri-bilimi/mlops)
2. [Derin Öğrenme Deployment](/topics/python/veri-bilimi/derin-ogrenme-deployment)
3. [Yapay Zeka Projeleri](/topics/python/veri-bilimi/yapay-zeka-projeleri)

## Faydalı Kaynaklar

- [Apache Spark Dokümantasyonu](https://spark.apache.org/docs/latest/)
- [Dask Dokümantasyonu](https://docs.dask.org/)
- [Apache Kafka Dokümantasyonu](https://kafka.apache.org/documentation/)
- [PySpark Örnekleri](https://spark.apache.org/examples.html)
`;

export default function BigDataProcessingPage() {
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