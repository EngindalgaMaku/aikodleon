import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Tekrarlayan Sinir Ağları (RNN) | Kodleon',
  description: 'Sıralı veri işleme ve zaman serisi analizi için RNN modellerini öğrenin.',
};

const content = `
# Tekrarlayan Sinir Ağları (RNN)

Tekrarlayan Sinir Ağları (RNN), sıralı verileri işlemek için tasarlanmış özel bir yapay sinir ağı türüdür. Metin, zaman serileri ve ses gibi sıralı verilerin analizi için idealdir.

## 1. RNN Mimarisi

RNN'ler, önceki adımların bilgisini saklayarak sıralı verileri işler. Temel RNN hücresi şu bileşenlerden oluşur:
- Giriş katmanı
- Gizli durum (hidden state)
- Çıkış katmanı

### Temel RNN Hücresi

\`\`\`python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length, input_size)
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_size)
        
        # RNN katmanı
        out, hidden = self.rnn(x, hidden)
        
        # Son zaman adımının çıktısı
        out = self.fc(out[:, -1, :])
        return out, hidden

# Model oluşturma
model = SimpleRNN(
    input_size=10,    # Giriş özellik sayısı
    hidden_size=20,   # Gizli katman boyutu
    output_size=2     # Çıkış sınıf sayısı
)
\`\`\`

## 2. LSTM (Long Short-Term Memory)

LSTM'ler, uzun vadeli bağımlılıkları öğrenmek için tasarlanmış özel bir RNN türüdür.

\`\`\`python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM katmanı
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, (hidden, cell) = self.lstm(x, (h0, c0))
        
        # Son zaman adımının çıktısı
        out = self.fc(out[:, -1, :])
        return out

# Model oluşturma
lstm_model = LSTMModel(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    output_size=1
)
\`\`\`

## 3. GRU (Gated Recurrent Unit)

GRU, LSTM'in daha basit bir versiyonudur ve benzer performans sunar.

\`\`\`python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, hidden = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Model oluşturma
gru_model = GRUModel(
    input_size=10,
    hidden_size=32,
    num_layers=2,
    output_size=1
)
\`\`\`

## 4. Seq2Seq Modeller

Seq2Seq modeller, bir diziyi başka bir diziye dönüştürmek için kullanılır (örn. makine çevirisi).

\`\`\`python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size)
        
        # Encoder
        _, hidden, cell = self.encoder(source)
        
        # İlk decoder girdisi
        decoder_input = target[:, 0:1]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t:t+1] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            decoder_input = target[:, t:t+1] if teacher_force else output
        
        return outputs
\`\`\`

## 5. Attention Mekanizması

Attention mekanizması, modelin giriş dizisinin farklı kısımlarına farklı ağırlıklar vermesini sağlar.

\`\`\`python
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # hidden shape: (batch_size, hidden_size)
        # encoder_outputs shape: (batch_size, seq_len, hidden_size)
        
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Attention skorları hesaplama
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        
        return F.softmax(attention, dim=1)
\`\`\`

## 6. Model Eğitimi ve Değerlendirme

\`\`\`python
def train_rnn(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')

def evaluate_rnn(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions.extend(output.numpy())
            actuals.extend(target.numpy())
    
    return np.array(predictions), np.array(actuals)
\`\`\`

## Alıştırmalar

1. **Temel RNN Uygulamaları**
   - Duygu analizi
   - Zaman serisi tahmini
   - Karakter seviyesi dil modeli

2. **LSTM ve GRU**
   - Metin sınıflandırma
   - Müzik üretimi
   - Hava durumu tahmini

3. **Seq2Seq ve Attention**
   - Makine çevirisi
   - Özetleme
   - Soru cevaplama

## Kaynaklar

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
`;

export default function RNNPage() {
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
        <Tabs defaultValue="basic">
          <TabsList>
            <TabsTrigger value="basic">Temel RNN</TabsTrigger>
            <TabsTrigger value="lstm">LSTM</TabsTrigger>
            <TabsTrigger value="seq2seq">Seq2Seq</TabsTrigger>
          </TabsList>
          
          <TabsContent value="basic">
            <Card>
              <CardHeader>
                <CardTitle>Temel RNN Örneği</CardTitle>
                <CardDescription>
                  Basit bir metin sınıflandırma uygulaması
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import torch
import torch.nn as nn

# Veri hazırlama
vocab_size = 1000
embedding_dim = 100
hidden_size = 64
num_classes = 2

class TextRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(output[:, -1, :])

# Model oluşturma ve eğitim
model = TextRNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Eğitim döngüsü
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="lstm">
            <Card>
              <CardHeader>
                <CardTitle>LSTM Örneği</CardTitle>
                <CardDescription>
                  Zaman serisi tahmini için LSTM modeli
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

# Model oluşturma
model = TimeSeriesLSTM(
    input_size=10,    # Özellik sayısı
    hidden_size=64,   # LSTM hücre boyutu
    num_layers=2      # LSTM katman sayısı
)

# Model eğitimi
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="seq2seq">
            <Card>
              <CardHeader>
                <CardTitle>Seq2Seq Örneği</CardTitle>
                <CardDescription>
                  Makine çevirisi için Seq2Seq model
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

# Model kullanımı
encoder = Encoder(src_vocab_size, embedding_dim, hidden_size)
decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_size)

def translate(sentence, src_vocab, tgt_vocab, max_length=50):
    model.eval()
    
    # Cümleyi sayılara dönüştür
    tokens = [src_vocab[word] for word in sentence.split()]
    source = torch.LongTensor(tokens).unsqueeze(0)
    
    # Encoder
    encoder_outputs, hidden, cell = encoder(source)
    
    # İlk decoder girdisi (BOS token)
    decoder_input = torch.tensor([tgt_vocab['<BOS>']])
    
    translation = []
    
    for _ in range(max_length):
        output, hidden, cell = decoder(decoder_input, hidden, cell)
        
        # En yüksek olasılıklı kelimeyi seç
        top1 = output.argmax(1)
        
        # EOS tokenı gelirse dur
        if top1.item() == tgt_vocab['<EOS>']:
            break
        
        translation.append(top1.item())
        decoder_input = top1
    
    # Sayıları kelimelere dönüştür
    return ' '.join([list(tgt_vocab.keys())[i] for i in translation])`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/derin-ogrenme/evrisimli-sinir-aglari">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Evrişimli Sinir Ağları
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/derin-ogrenme/modern-mimariler">
            Sonraki Konu: Modern Mimari ve Yaklaşımlar
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Button>
      </div>
      
      <div className="mt-16 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Kodleon | Yapay Zeka Eğitim Platformu</p>
      </div>
    </div>
  );
} 