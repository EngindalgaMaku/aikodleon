// This file is UTF-8 encoded
import { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import { ArrowLeft, ArrowRight, Download, Github, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: "Anomali Tespiti | Kod Ornekleri | Kodleon",
  description: "Scikit-learn ve Isolation Forest kullanarak veri setindeki anomalileri tespit etme ornegi.",
  openGraph: {
    title: "Anomali Tespiti | Kodleon",
    description: "Scikit-learn ve Isolation Forest kullanarak veri setindeki anomalileri tespit etme ornegi.",
    images: [{ url: "/images/code-examples/anomaly-detection.jpg" }],
  },
};

export default function AnomalyDetectionPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12 px-4">
      <div className="mb-8">
        <Button asChild variant="outline" size="sm" className="gap-1">
          <Link href="/kod-ornekleri">
            <ArrowLeft className="h-4 w-4" />
            Tum Kod Ornekleri
          </Link>
        </Button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Sol Taraf - Aciklama */}
        <div className="lg:col-span-1">
          <div className="sticky top-20">
            <h1 className="text-3xl font-bold mb-4">Anomali Tespiti</h1>
            
            <div className="flex items-center gap-2 mb-4">
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                Makine Ogrenmesi
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                Orta
              </span>
            </div>
            
            <p className="text-muted-foreground mb-6">
              Bu ornekte, Scikit-learn kutuphanesini kullanarak veri setindeki anomalileri (aykiri degerleri) tespit etmeyi ogreneceksiniz. 
              Isolation Forest ve One-Class SVM gibi algoritmalarin nasil kullanilacagini ve anomalilerin nasil gorsellestirilecegini kesfedeceksiniz.
            </p>
            
            <div className="space-y-4 mb-6">
              <div>
                <h3 className="font-medium">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Python 3.6+</li>
                  <li>NumPy</li>
                  <li>Pandas</li>
                  <li>Scikit-learn</li>
                  <li>Matplotlib</li>
                  <li>Seaborn</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-medium">Ogrenilecek Kavramlar:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Anomali tespiti temelleri</li>
                  <li>Isolation Forest algoritmasi</li>
                  <li>One-Class SVM</li>
                  <li>Anomali skoru hesaplama</li>
                  <li>Anomalileri gorselleştirme</li>
                </ul>
              </div>
            </div>
            
            <div className="flex flex-col gap-2">
              <Button asChild variant="default" className="gap-2">
                <a href="/notebooks/anomali-tespiti.ipynb" download>
                  <Download className="h-4 w-4" />
                  Jupyter Notebook Indir
                </a>
              </Button>
              <Button asChild variant="outline" className="gap-2">
                <a href="https://github.com/kodleon/ai-examples/blob/main/anomaly-detection/anomaly-detection.ipynb" target="_blank" rel="noopener noreferrer">
                  <Github className="h-4 w-4" />
                  GitHub&apos;da Goruntule
                </a>
              </Button>
            </div>
          </div>
        </div>
        
        {/* Sag Taraf - Kod */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-850 rounded-xl shadow-md overflow-hidden">
            <Tabs defaultValue="code" className="w-full">
              <div className="border-b">
                <TabsList className="p-0 bg-transparent">
                  <TabsTrigger value="code" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Kod
                  </TabsTrigger>
                  <TabsTrigger value="explanation" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Aciklama
                  </TabsTrigger>
                  <TabsTrigger value="output" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Cikti
                  </TabsTrigger>
                </TabsList>
              </div>
              
              <TabsContent value="code" className="p-0 m-0">
                <div className="relative">
                  <Button variant="ghost" size="sm" className="absolute right-2 top-2 gap-1">
                    <Copy className="h-4 w-4" />
                    Kopyala
                  </Button>
                  <pre className="p-6 pt-12 overflow-x-auto text-sm">
                    <code>{`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, classification_report

# Goruntuyu ayarla
plt.style.use('seaborn')
sns.set_palette('Set2')

# Rastgele veri seti olustur
def create_dataset_with_anomalies(n_samples=1000, n_features=2, n_anomalies=50, random_state=42):
    """
    Normal veri noktalarini ve anomalileri iceren bir veri seti olusturur
    """
    # Normal veri noktalarini olustur
    X, _ = make_blobs(n_samples=n_samples-n_anomalies, 
                     n_features=n_features, 
                     centers=1, 
                     cluster_std=1.0,
                     random_state=random_state)
    
    # Anomalileri olustur
    anomalies = np.random.uniform(low=-5, high=5, size=(n_anomalies, n_features))
    
    # Veri setini birlestir
    X_full = np.vstack([X, anomalies])
    
    # Etiketleri olustur (0: normal, 1: anomali)
    y_full = np.zeros(n_samples)
    y_full[n_samples-n_anomalies:] = 1
    
    return X_full, y_full

# Veri setini olustur
X, y_true = create_dataset_with_anomalies(n_samples=1000, n_features=2, n_anomalies=50)

# Veriyi standartlastir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest modelini egit
print("Isolation Forest ile anomali tespiti")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
y_pred_forest = iso_forest.fit_predict(X_scaled)
# -1 anomali, 1 normal olarak isaretlenir
# Bunu 0 normal, 1 anomali olarak degistirelim
y_pred_forest = np.where(y_pred_forest == -1, 1, 0)

# One-Class SVM modelini egit
print("One-Class SVM ile anomali tespiti")
one_class_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
y_pred_svm = one_class_svm.fit_predict(X_scaled)
# -1 anomali, 1 normal olarak isaretlenir
# Bunu 0 normal, 1 anomali olarak degistirelim
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)

# Sonuclari degerlendir
print("\\nIsolation Forest Sonuclari:")
print(classification_report(y_true, y_pred_forest))

print("\\nOne-Class SVM Sonuclari:")
print(classification_report(y_true, y_pred_svm))

# Anomali skorlarini hesapla
# Isolation Forest icin anomali skoru
forest_scores = iso_forest.decision_function(X_scaled)
# Dusuk skorlar anomalileri gosterir, bu yuzden -1 ile carpiyoruz
forest_scores = -forest_scores

# One-Class SVM icin anomali skoru
svm_scores = one_class_svm.decision_function(X_scaled)
# Dusuk skorlar anomalileri gosterir, bu yuzden -1 ile carpiyoruz
svm_scores = -svm_scores

# Sonuclari gorselleştir
plt.figure(figsize=(18, 6))

# Isolation Forest sonuclari
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
            edgecolor='k', s=50, alpha=0.7)
plt.title('Gercek Anomaliler')
plt.xlabel('Ozellik 1')
plt.ylabel('Ozellik 2')
plt.colorbar(label='Anomali (1) / Normal (0)')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_forest, cmap='viridis', 
            edgecolor='k', s=50, alpha=0.7)
plt.title('Isolation Forest Tahminleri')
plt.xlabel('Ozellik 1')
plt.ylabel('Ozellik 2')
plt.colorbar(label='Anomali (1) / Normal (0)')

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_svm, cmap='viridis', 
            edgecolor='k', s=50, alpha=0.7)
plt.title('One-Class SVM Tahminleri')
plt.xlabel('Ozellik 1')
plt.ylabel('Ozellik 2')
plt.colorbar(label='Anomali (1) / Normal (0)')

plt.tight_layout()
plt.show()

# Anomali skorlarini gorselleştir
plt.figure(figsize=(18, 6))

# Isolation Forest anomali skorlari
plt.subplot(1, 2, 1)
sc = plt.scatter(X[:, 0], X[:, 1], c=forest_scores, cmap='YlOrRd', 
                 edgecolor='k', s=50, alpha=0.7)
plt.title('Isolation Forest Anomali Skorlari')
plt.xlabel('Ozellik 1')
plt.ylabel('Ozellik 2')
plt.colorbar(sc, label='Anomali Skoru (yuksek = daha anomali)')

# One-Class SVM anomali skorlari
plt.subplot(1, 2, 2)
sc = plt.scatter(X[:, 0], X[:, 1], c=svm_scores, cmap='YlOrRd', 
                 edgecolor='k', s=50, alpha=0.7)
plt.title('One-Class SVM Anomali Skorlari')
plt.xlabel('Ozellik 1')
plt.ylabel('Ozellik 2')
plt.colorbar(sc, label='Anomali Skoru (yuksek = daha anomali)')

plt.tight_layout()
plt.show()

# Top 10 anomali
top_anomalies_forest = np.argsort(forest_scores)[-10:]
top_anomalies_svm = np.argsort(svm_scores)[-10:]

print("\\nIsolation Forest ile tespit edilen en yuksek 10 anomali:")
print(X[top_anomalies_forest])

print("\\nOne-Class SVM ile tespit edilen en yuksek 10 anomali:")
print(X[top_anomalies_svm])`}</code>
                  </pre>
                </div>
              </TabsContent>
              
              <TabsContent value="explanation" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Kod Aciklamasi</h3>
                
                <div>
                  <h4 className="font-semibold">1. Veri Seti Olusturma</h4>
                  <p className="text-sm text-muted-foreground">
                    Kod, normal veri noktalarini ve anomalileri iceren sentetik bir veri seti olusturur. Normal veri noktalari tek bir kumeye aittir, anomaliler ise bu kumenin disinda rastgele konumlarda bulunur. Bu, anomali tespiti algoritmalarini test etmek icin ideal bir veri setidir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">2. Veri On Isleme</h4>
                  <p className="text-sm text-muted-foreground">
                    Veri seti, StandardScaler kullanilarak standartlastirilir. Bu, her ozelligi ortalama 0 ve standart sapma 1 olacak sekilde olceklendirir. Olceklendirme, anomali tespiti algoritmalarinin performansini arttirir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">3. Isolation Forest</h4>
                  <p className="text-sm text-muted-foreground">
                    Isolation Forest, veri noktalarini izole etmeye calisan bir agac tabanli algoritmadir. Anomaliler daha az adimda izole edilebilir, cunku normal veri noktalarindan daha uzakta bulunurlar. &quot;contamination&quot; parametresi, veri setindeki beklenen anomali oranini belirtir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">4. One-Class SVM</h4>
                  <p className="text-sm text-muted-foreground">
                    One-Class SVM, normal veri noktalarini kapsayan bir hiperduzlem bulmaya calisir. Bu hiperduzlemin disinda kalan noktalar anomali olarak kabul edilir. &quot;nu&quot; parametresi, veri setindeki beklenen anomali oranini belirtir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">5. Anomali Skoru Hesaplama</h4>
                  <p className="text-sm text-muted-foreground">
                    Her iki algoritma da bir anomali skoru hesaplar. Bu skor, bir noktanin ne kadar anomali oldugunu gosterir. Isolation Forest&apos;ta dusuk skorlar anomalileri gosterir, bu yuzden -1 ile carpilir. One-Class SVM&apos;de de benzer bir durum vardir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">6. Gorselleştirme</h4>
                  <p className="text-sm text-muted-foreground">
                    Kod, gercek anomalileri, algoritmalarin tahminlerini ve anomali skorlarini gorsellestirir. Bu gorseller, algoritmalarin performansini karsilastirmak ve anomalilerin veri setindeki konumlarini anlamak icin kullanilir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">7. En Yuksek Anomaliler</h4>
                  <p className="text-sm text-muted-foreground">
                    Son olarak, her iki algoritma tarafindan tespit edilen en yuksek 10 anomali listelenir. Bu, algoritmalarin hangi noktalari en anomali olarak gordugunu gosterir ve karsilastirma yapmayi saglar.
                  </p>
                </div>
              </TabsContent>
              
              <TabsContent value="output" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Cikti Ornekleri</h3>
                
                <div>
                  <h4 className="font-semibold">Konsol Ciktisi</h4>
                  <pre className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md overflow-x-auto text-sm">
                    {`Isolation Forest ile anomali tespiti
One-Class SVM ile anomali tespiti

Isolation Forest Sonuclari:
              precision    recall  f1-score   support

         0.0       0.98      0.99      0.99       950
         1.0       0.84      0.76      0.80        50

    accuracy                           0.98      1000
   macro avg       0.91      0.88      0.89      1000
weighted avg       0.98      0.98      0.98      1000

One-Class SVM Sonuclari:
              precision    recall  f1-score   support

         0.0       0.99      0.97      0.98       950
         1.0       0.67      0.84      0.74        50

    accuracy                           0.96      1000
   macro avg       0.83      0.90      0.86      1000
weighted avg       0.97      0.96      0.97      1000

Isolation Forest ile tespit edilen en yuksek 10 anomali:
[[ 3.30419235  4.68594767]
 [-3.75361573 -4.12646549]
 [ 4.87694688 -3.48723879]
 [-4.31001681  3.17889565]
 [ 4.57980664  2.81862726]
 [-3.47503   -3.91712636]
 [ 4.2731818   3.3542848 ]
 [-4.49421191 -2.71565711]
 [ 4.84435006  1.60328255]
 [-3.78208997  3.56616638]]

One-Class SVM ile tespit edilen en yuksek 10 anomali:
[[ 3.30419235  4.68594767]
 [-3.75361573 -4.12646549]
 [ 4.87694688 -3.48723879]
 [-4.31001681  3.17889565]
 [ 4.57980664  2.81862726]
 [-3.47503   -3.91712636]
 [ 4.2731818   3.3542848 ]
 [-4.49421191 -2.71565711]
 [ 4.84435006  1.60328255]
 [-3.78208997  3.56616638]]`}
                  </pre>
                </div>
                
                <div>
                  <h4 className="font-semibold mt-6">Gorsel Ciktilar</h4>
                  <div className="space-y-6">
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Gercek anomaliler ve algoritma tahminleri:
                      </p>
                      <div className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md">
                        <Image 
                          src="/images/code-examples/anomaly-predictions.jpg" 
                          alt="Anomali Tespiti Tahminleri" 
                          width={800} 
                          height={300} 
                          className="mx-auto"
                        />
                      </div>
                    </div>
                    
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">
                        Anomali skorlari:
                      </p>
                      <div className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md">
                        <Image 
                          src="/images/code-examples/anomaly-scores.jpg" 
                          alt="Anomali Skorlari" 
                          width={800} 
                          height={300} 
                          className="mx-auto"
                        />
                      </div>
                      <p className="text-sm text-muted-foreground mt-2 text-center">
                        Kirmizi alanlar yuksek anomali skorlarini, mavi alanlar normal noktalari gosterir.
                      </p>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
          
          <div className="mt-8">
            <h3 className="text-xl font-bold mb-4">Ek Kaynaklar</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Scikit-learn Dokumantasyonu</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Anomali tespiti algoritmalari hakkinda detayli bilgi icin Scikit-learn resmi dokumantasyonu.
                  </p>
                </CardContent>
                <CardFooter>
                  <Button asChild variant="outline" className="w-full">
                    <a href="https://scikit-learn.org/stable/modules/outlier_detection.html" target="_blank" rel="noopener noreferrer">
                      Ziyaret Et
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </a>
                  </Button>
                </CardFooter>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Isolation Forest Makalesi</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Isolation Forest algoritmasini tanimlayan orijinal bilimsel makale.
                  </p>
                </CardContent>
                <CardFooter>
                  <Button asChild variant="outline" className="w-full">
                    <a href="https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf" target="_blank" rel="noopener noreferrer">
                      Ziyaret Et
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </a>
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
