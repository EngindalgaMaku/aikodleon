import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Derin Öğrenme Pratik Uygulamalar | Kodleon',
  description: 'Görüntü işleme, doğal dil işleme, ses tanıma ve daha fazlası için pratik derin öğrenme uygulamaları.',
};

const content = `
# Derin Öğrenme Pratik Uygulamalar

Bu bölümde, derin öğrenmenin gerçek dünya problemlerine nasıl uygulanacağını öğreneceksiniz. Her uygulama için detaylı kod örnekleri ve açıklamalar sunulmuştur.

## 1. Görüntü Sınıflandırma

ResNet kullanarak çok sınıflı görüntü sınıflandırma uygulaması.

\`\`\`python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

class ImageClassifier:
    def __init__(self, num_classes):
        # ResNet50 modelini yükle
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Son katmanı değiştir
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Veri dönüşümleri
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def train(self, train_loader, num_epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    
    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = self.transform(image).unsqueeze(0)
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

# Kullanım örneği
classifier = ImageClassifier(num_classes=10)
classifier.train(train_loader)
prediction = classifier.predict(test_image)
\`\`\`

## 2. Doğal Dil İşleme

BERT kullanarak metin sınıflandırma uygulaması.

\`\`\`python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class TextClassifier:
    def __init__(self, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        )
    
    def prepare_input(self, text):
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    
    def train(self, train_texts, train_labels, num_epochs=3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for text, label in zip(train_texts, train_labels):
                inputs = self.prepare_input(text)
                outputs = self.model(**inputs, labels=label)
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(train_texts)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            inputs = self.prepare_input(text)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            return predictions.item()

# Kullanım örneği
classifier = TextClassifier(num_labels=2)
classifier.train(train_texts, train_labels)
sentiment = classifier.predict("This movie was great!")
\`\`\`

## 3. Ses Tanıma

Wav2Vec2 kullanarak konuşma tanıma uygulaması.

\`\`\`python
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

class SpeechRecognizer:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    def load_audio(self, file_path):
        # Ses dosyasını yükle ve örnekleme hızını ayarla
        speech, sr = librosa.load(file_path, sr=16000)
        return speech
    
    def process_audio(self, speech):
        # Ses verisini model girdisine dönüştür
        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        return inputs
    
    def transcribe(self, file_path):
        # Ses dosyasını metne dönüştür
        speech = self.load_audio(file_path)
        inputs = self.process_audio(speech)
        
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)
        
        return transcription[0]

# Kullanım örneği
recognizer = SpeechRecognizer()
text = recognizer.transcribe("audio.wav")
print(f"Transcription: {text}")
\`\`\`

## 4. Öneri Sistemleri

Matrix Factorization kullanarak öneri sistemi uygulaması.

\`\`\`python
import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=100):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Başlangıç değerlerini normalize et
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        return torch.sum(user_embeds * item_embeds, dim=1)

class RecommenderSystem:
    def __init__(self, num_users, num_items, embedding_dim=100):
        self.model = MatrixFactorization(num_users, num_items, embedding_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
    
    def train(self, train_loader, num_epochs=10):
        for epoch in range(num_epochs):
            total_loss = 0
            for user_ids, item_ids, ratings in train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    def recommend(self, user_id, top_k=5):
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id])
            all_items = torch.arange(self.model.item_embeddings.num_embeddings)
            predictions = self.model(
                user_tensor.repeat(len(all_items)),
                all_items
            )
            top_items = torch.topk(predictions, k=top_k).indices
            return top_items.tolist()

# Kullanım örneği
recommender = RecommenderSystem(num_users=1000, num_items=500)
recommender.train(train_loader)
recommendations = recommender.recommend(user_id=42, top_k=5)
\`\`\`

## 5. Anomali Tespiti

Autoencoder kullanarak anomali tespiti uygulaması.

\`\`\`python
class AnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetection:
    def __init__(self, input_dim, threshold=0.1):
        self.model = AnomalyDetector(input_dim)
        self.threshold = threshold
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss(reduction='none')
    
    def train(self, train_loader, num_epochs=50):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for data in train_loader:
                self.optimizer.zero_grad()
                reconstructed = self.model(data)
                loss = self.criterion(reconstructed, data).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    def detect(self, data):
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data)
            reconstruction_error = torch.mean(self.criterion(reconstructed, data), dim=1)
            return reconstruction_error > self.threshold

# Kullanım örneği
detector = AnomalyDetection(input_dim=100)
detector.train(normal_data_loader)
anomalies = detector.detect(test_data)
\`\`\`

## Alıştırmalar

1. **Görüntü İşleme**
   - Kedi/köpek sınıflandırma modeli
   - Yüz tanıma sistemi
   - Nesne tespit uygulaması

2. **Doğal Dil İşleme**
   - Spam filtreleme
   - Duygu analizi
   - Metin özetleme

3. **Ses İşleme**
   - Konuşmacı tanıma
   - Müzik türü sınıflandırma
   - Ses filtreleme

4. **Öneri Sistemleri**
   - Film önerisi
   - Ürün önerisi
   - İçerik önerisi

5. **Anomali Tespiti**
   - Kredi kartı dolandırıcılığı tespiti
   - Ağ trafiği anomali tespiti
   - Sensör verisi anomali tespiti

## Kaynaklar

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
`;

export default function PracticalApplicationsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/python/derin-ogrenme">
            <ArrowLeft className="h-4 w-4" />
            Derin Öğrenmeye Dön
          </Link>
        </Button>
      </div>
      
      <div className="prose prose-lg dark:prose-invert max-w-none">
        <MarkdownContent content={content} />
      </div>
      
      {/* Interactive Examples */}
      <div className="my-12">
        <h2 className="text-3xl font-bold mb-8">İnteraktif Örnekler</h2>
        <Tabs defaultValue="image">
          <TabsList>
            <TabsTrigger value="image">Görüntü İşleme</TabsTrigger>
            <TabsTrigger value="nlp">Doğal Dil İşleme</TabsTrigger>
            <TabsTrigger value="recommender">Öneri Sistemi</TabsTrigger>
          </TabsList>
          
          <TabsContent value="image">
            <Card>
              <CardHeader>
                <CardTitle>ResNet ile Görüntü Sınıflandırma</CardTitle>
                <CardDescription>
                  Transfer learning kullanarak özel veri seti üzerinde eğitim
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

def train_classifier(train_dir, num_classes, num_epochs=10):
    # Model ve dönüşümler
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
    ])
    
    # Veri yükleyici
    dataset = ImageFolder(train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Eğitim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}')
    
    return model

# Model eğitimi
model = train_classifier('dataset/train', num_classes=10)

# Tahmin
def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="nlp">
            <Card>
              <CardHeader>
                <CardTitle>BERT ile Duygu Analizi</CardTitle>
                <CardDescription>
                  Hugging Face transformers kullanarak duygu analizi
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

def setup_sentiment_analyzer():
    # Model ve tokenizer yükleme
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    return model, tokenizer

def analyze_sentiment(text, model, tokenizer):
    # Metni tokenize et
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Tahmin
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    sentiment = "Positive" if prediction.item() == 1 else "Negative"
    confidence = probabilities[0][prediction.item()].item()
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': {
            'negative': probabilities[0][0].item(),
            'positive': probabilities[0][1].item()
        }
    }

# Kullanım
model, tokenizer = setup_sentiment_analyzer()
result = analyze_sentiment(
    "This movie was absolutely fantastic!",
    model,
    tokenizer
)
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Probabilities: {result['probabilities']}")`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="recommender">
            <Card>
              <CardHeader>
                <CardTitle>Collaborative Filtering Öneri Sistemi</CardTitle>
                <CardDescription>
                  Matrix Factorization tabanlı film önerisi
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class MovieRecommender:
    def __init__(self, num_users, num_movies, embedding_dim=100):
        self.model = nn.ModuleDict({
            'user_embeddings': nn.Embedding(num_users, embedding_dim),
            'movie_embeddings': nn.Embedding(num_movies, embedding_dim),
            'user_biases': nn.Embedding(num_users, 1),
            'movie_biases': nn.Embedding(num_movies, 1)
        })
        
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def forward(self, user_ids, movie_ids):
        # Embeddingler
        user_embeds = self.model['user_embeddings'](user_ids)
        movie_embeds = self.model['movie_embeddings'](movie_ids)
        
        # Biaslar
        user_bias = self.model['user_biases'](user_ids).squeeze()
        movie_bias = self.model['movie_biases'](movie_ids).squeeze()
        
        # Tahmin
        dot_products = torch.sum(user_embeds * movie_embeds, dim=1)
        return dot_products + user_bias + movie_bias
    
    def train(self, train_loader, num_epochs=10):
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for user_ids, movie_ids, ratings in train_loader:
                self.optimizer.zero_grad()
                
                predictions = self.forward(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    def recommend_movies(self, user_id, n_recommendations=5):
        with torch.no_grad():
            # Tüm filmler için tahmin yap
            user_tensor = torch.tensor([user_id])
            all_movies = torch.arange(self.model['movie_embeddings'].num_embeddings)
            
            predictions = self.forward(
                user_tensor.repeat(len(all_movies)),
                all_movies
            )
            
            # En yüksek puanlı filmleri seç
            top_movies = torch.topk(predictions, k=n_recommendations)
            return top_movies.indices.tolist()

# Kullanım
recommender = MovieRecommender(num_users=1000, num_movies=2000)
recommender.train(train_loader)

# Kullanıcı için film önerisi
user_id = 42
recommendations = recommender.recommend_movies(user_id)
print(f"Recommended movies for user {user_id}: {recommendations}")`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 