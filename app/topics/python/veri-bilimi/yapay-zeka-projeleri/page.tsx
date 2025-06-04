import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Yapay Zeka Projeleri | Python Veri Bilimi | Kodleon',
  description: 'Görüntü işleme, doğal dil işleme ve pekiştirmeli öğrenme projeleri ile pratik yapay zeka uygulamaları.',
};

const content = `
# Yapay Zeka Projeleri

Bu bölümde, farklı yapay zeka alanlarında pratik projeler geliştireceğiz. Her proje, gerçek dünya problemlerine yönelik çözümler sunmaktadır.

## Görüntü İşleme Projesi: Nesne Takip Sistemi

\`\`\`python
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import torch
import torchvision.transforms as transforms

class NesneTakipSistemi:
    def __init__(self, model_turu="mobilenet"):
        self.model_turu = model_turu
        
        if model_turu == "mobilenet":
            self.model = MobileNetV2(weights='imagenet')
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            
        # Nesne takip için
        self.tracker = cv2.TrackerCSRT_create()
        self.bbox = None
        self.tracking = False
        
    def goruntu_on_isleme(self, frame):
        if self.model_turu == "mobilenet":
            # MobileNet için ön işleme
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess_input(frame)
            return np.expand_dims(frame, axis=0)
        else:
            # YOLOv5 için ön işleme
            return frame
            
    def nesne_tespit(self, frame):
        isle_frame = self.goruntu_on_isleme(frame)
        
        if self.model_turu == "mobilenet":
            tahminler = self.model.predict(isle_frame)
            sinif_idx = np.argmax(tahminler)
            guven = tahminler[0][sinif_idx]
            return sinif_idx, guven
        else:
            sonuclar = self.model(frame)
            return sonuclar.xyxy[0].cpu().numpy()
            
    def takip_baslat(self, frame, bbox):
        self.bbox = bbox
        self.tracking = True
        self.tracker.init(frame, bbox)
        
    def takip_guncelle(self, frame):
        if self.tracking:
            success, bbox = self.tracker.update(frame)
            if success:
                self.bbox = bbox
                return True, bbox
            else:
                self.tracking = False
                return False, None
        return False, None
        
    def goruntu_isle(self, frame):
        # Nesne tespiti
        if not self.tracking:
            if self.model_turu == "mobilenet":
                sinif_idx, guven = self.nesne_tespit(frame)
                if guven > 0.5:
                    # Basit bir bbox tahmini
                    h, w = frame.shape[:2]
                    self.bbox = (w//4, h//4, w//2, h//2)
                    self.takip_baslat(frame, self.bbox)
            else:
                tespitler = self.nesne_tespit(frame)
                if len(tespitler) > 0:
                    # En yüksek güvenilirlikli tespit
                    tespit = tespitler[0]
                    self.bbox = (
                        int(tespit[0]), int(tespit[1]),
                        int(tespit[2]-tespit[0]),
                        int(tespit[3]-tespit[1])
                    )
                    self.takip_baslat(frame, self.bbox)
                    
        # Nesne takibi
        success, bbox = self.takip_guncelle(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return frame

# Kullanım örneği
if __name__ == "__main__":
    takip_sistemi = NesneTakipSistemi(model_turu="yolov5")
    
    # Video yakalama
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Görüntü işleme
        frame = takip_sistemi.goruntu_isle(frame)
        
        # Göster
        cv2.imshow('Nesne Takip', frame)
        
        # Çıkış için 'q' tuşu
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
\`\`\`

## Doğal Dil İşleme Projesi: Duygu Analizi Sistemi

\`\`\`python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class DuyguAnaliziSistemi:
    def __init__(self, model_adi="dbmdz/bert-base-turkish-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_adi)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_adi,
            num_labels=3  # pozitif, negatif, nötr
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def veri_hazirla(self, metinler, etiketler=None):
        # Tokenization
        tokenized = self.tokenizer(
            metinler,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        if etiketler is not None:
            return DuyguVeriSeti(tokenized, etiketler)
        return tokenized
        
    def model_egit(self, egitim_veri, dogrulama_veri, epochs=3):
        # Eğitim parametreleri
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Eğitim döngüsü
        self.model.train()
        for epoch in range(epochs):
            toplam_kayip = 0
            for batch in egitim_veri:
                # Batch'i GPU'ya taşı
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                kayip = outputs.loss
                toplam_kayip += kayip.item()
                
                # Backward pass
                kayip.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            # Doğrulama
            dogruluk = self.model_degerlendir(dogrulama_veri)
            print(f"Epoch {epoch+1}: Kayıp = {toplam_kayip/len(egitim_veri):.4f}, Doğruluk = {dogruluk:.4f}")
            
    def model_degerlendir(self, test_veri):
        self.model.eval()
        dogru = 0
        toplam = 0
        
        with torch.no_grad():
            for batch in test_veri:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                tahminler = torch.argmax(outputs.logits, dim=1)
                dogru += (tahminler == labels).sum().item()
                toplam += labels.size(0)
                
        return dogru / toplam
        
    def duygu_analiz(self, metin):
        # Metni tokenize et
        inputs = self.tokenizer(
            metin,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Tahmin yap
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            tahminler = F.softmax(outputs.logits, dim=1)
            
        # Sonuçları yorumla
        etiketler = ['Negatif', 'Nötr', 'Pozitif']
        sonuclar = {
            etiket: float(tahmin)
            for etiket, tahmin in zip(etiketler, tahminler[0])
        }
        
        return sonuclar

class DuyguVeriSeti(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Kullanım örneği
if __name__ == "__main__":
    # Sistem oluştur
    duygu_sistemi = DuyguAnaliziSistemi()
    
    # Örnek veri
    metinler = [
        "Bu ürün gerçekten harika!",
        "Hiç beğenmedim, çok kötü.",
        "Fiyatı biraz yüksek ama kaliteli.",
        "Eh işte, idare eder."
    ]
    etiketler = [2, 0, 1, 1]  # 2:pozitif, 0:negatif, 1:nötr
    
    # Veri hazırlama
    veri = duygu_sistemi.veri_hazirla(metinler, etiketler)
    train_loader = DataLoader(veri, batch_size=2, shuffle=True)
    
    # Model eğitimi
    duygu_sistemi.model_egit(train_loader, train_loader)
    
    # Yeni metin analizi
    yeni_metin = "Bu ürünü herkese tavsiye ederim!"
    sonuc = duygu_sistemi.duygu_analiz(yeni_metin)
    print(f"Metin: {yeni_metin}")
    print("Duygu Analizi Sonuçları:", sonuc)
\`\`\`

## Pekiştirmeli Öğrenme Projesi: Oyun Oynama Ajanı

\`\`\`python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAjan:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
        
    def hafiza_kaydet(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def eylem_sec(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()
            
    def batch_egit(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch]).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in minibatch]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def target_model_guncelle(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def model_kaydet(self, dosya_yolu):
        torch.save(self.model.state_dict(), dosya_yolu)
        
    def model_yukle(self, dosya_yolu):
        self.model.load_state_dict(torch.load(dosya_yolu))
        self.target_model.load_state_dict(self.model.state_dict())

# Kullanım örneği
if __name__ == "__main__":
    # Oyun ortamı
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Ajan oluştur
    ajan = DQNAjan(state_size, action_size)
    
    # Eğitim parametreleri
    episodes = 1000
    batch_size = 32
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for time in range(500):
            # Eylem seç ve uygula
            action = ajan.eylem_sec(state)
            next_state, reward, done, _ = env.step(action)
            
            # Hafızaya kaydet
            ajan.hafiza_kaydet(state, action, reward, next_state, done)
            
            # Batch eğitimi
            ajan.batch_egit(batch_size)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        # Her 10 bölümde bir target model güncelle
        if episode % 10 == 0:
            ajan.target_model_guncelle()
            
        print(f"Episode: {episode+1}, Toplam Ödül: {total_reward}")
        
        # Başarı kriteri
        if total_reward >= 475:
            print("Ortam çözüldü!")
            ajan.model_kaydet("cartpole_model.pth")
            break
            
    env.close()
\`\`\`

## Alıştırmalar

1. **Görüntü İşleme**
   - Farklı nesne tespit modellerini deneyin
   - Çoklu nesne takibi ekleyin
   - Hareket tahmini özelliği ekleyin

2. **Doğal Dil İşleme**
   - Çok dilli duygu analizi yapın
   - Konu sınıflandırma ekleyin
   - Model performansını iyileştirin

3. **Pekiştirmeli Öğrenme**
   - Farklı oyun ortamlarında test edin
   - Prioritized Experience Replay ekleyin
   - A3C algoritmasını implemente edin

## Sonraki Adımlar

1. [MLOps ve DevOps](/topics/python/veri-bilimi/mlops)
2. [Derin Öğrenme Deployment](/topics/python/veri-bilimi/derin-ogrenme-deployment)
3. [Büyük Veri İşleme](/topics/python/veri-bilimi/buyuk-veri)

## Faydalı Kaynaklar

- [OpenCV Dokümantasyonu](https://docs.opencv.org/)
- [Hugging Face Transformers](https://huggingface.co/docs)
- [OpenAI Gym Dokümantasyonu](https://gym.openai.com/docs/)
- [PyTorch Dokümantasyonu](https://pytorch.org/docs/)
`;

export default function AIProjectsPage() {
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