import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: 'Modern Derin Öğrenme Mimarileri | Kodleon',
  description: 'Transformer, GAN, AutoEncoder gibi modern derin öğrenme mimarilerini ve tekniklerini öğrenin.',
};

const content = `
# Modern Derin Öğrenme Mimarileri

Modern derin öğrenme mimarileri, klasik yapay sinir ağlarının ötesine geçerek daha karmaşık problemleri çözebilen ve daha etkili öğrenme gerçekleştirebilen yapılardır.

## 1. Transformer Mimarisi

Transformer mimarisi, doğal dil işleme alanında devrim yaratan ve self-attention mekanizmasını kullanan bir mimaridir.

### Temel Transformer Yapısı

\`\`\`python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        # Linear projections and split into heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        
        # Reshape and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        
        # Feed forward
        ff = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff))
\`\`\`

## 2. GANs (Üretici Çekişmeli Ağlar)

GANs, gerçekçi veri örnekleri üretebilen iki ağın (üretici ve ayırıcı) birbirleriyle rekabet ettiği bir mimaridir.

\`\`\`python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# GAN Eğitimi
def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim):
    adversarial_loss = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Gerçek ve sahte etiketler
            valid = torch.ones(imgs.size(0), 1)
            fake = torch.zeros(imgs.size(0), 1)
            
            # Gerçek görüntülerle discriminator eğitimi
            real_loss = adversarial_loss(discriminator(imgs), valid)
            
            # Sahte görüntülerle discriminator eğitimi
            z = torch.randn(imgs.size(0), latent_dim)
            fake_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # Generator eğitimi
            g_loss = adversarial_loss(discriminator(fake_imgs), valid)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
\`\`\`

## 3. AutoEncoder'lar

AutoEncoder'lar, veriyi sıkıştırmayı ve yeniden oluşturmayı öğrenen yapılardır.

\`\`\`python
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
\`\`\`

## 4. Self-Supervised Learning

Self-supervised learning, etiketlenmemiş verilerden anlamlı temsiller öğrenmeyi amaçlar.

\`\`\`python
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        
        # Encoder network (örn. ResNet)
        self.encoder = base_encoder
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.fc.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return nn.functional.normalize(z, dim=1)

def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / temperature)
    
    mask = torch.zeros_like(sim_matrix)
    mask[range(batch_size), range(batch_size, 2*batch_size)] = 1.
    mask[range(batch_size, 2*batch_size), range(batch_size)] = 1.
    
    sim_matrix = sim_matrix * (1 - torch.eye(2*batch_size).to(z.device))
    
    loss = -torch.log(
        sim_matrix / (sim_matrix.sum(dim=1).view(-1, 1) + 1e-8)
    )
    loss = (loss * mask).sum() / mask.sum()
    return loss
\`\`\`

## 5. Few-Shot Learning

Few-shot learning, az sayıda örnekle yeni görevleri öğrenmeyi amaçlar.

\`\`\`python
class PrototypicalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, support_set, query_set, n_way, n_shot):
        # Support set encoding
        support = self.encoder(support_set)
        support = support.view(n_way, n_shot, -1)
        prototypes = support.mean(dim=1)  # Class prototypes
        
        # Query set encoding
        queries = self.encoder(query_set)
        
        # Calculate distances
        dists = torch.cdist(queries, prototypes)
        return -dists  # Return negative distances as logits
\`\`\`

## Alıştırmalar

1. **Transformer Uygulamaları**
   - Metin sınıflandırma
   - Makine çevirisi
   - Duygu analizi

2. **GAN Projeleri**
   - Yüz üretimi
   - Stil transferi
   - Görüntü süper çözünürlük

3. **AutoEncoder Deneyleri**
   - Gürültü giderme
   - Anomali tespiti
   - Özellik çıkarımı

## Kaynaklar

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GAN Tutorial](https://arxiv.org/abs/1701.00160)
- [Self-Supervised Learning](https://arxiv.org/abs/2002.05709)
`;

export default function ModernArchitecturesPage() {
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
        <Tabs defaultValue="transformer">
          <TabsList>
            <TabsTrigger value="transformer">Transformer</TabsTrigger>
            <TabsTrigger value="gan">GAN</TabsTrigger>
            <TabsTrigger value="autoencoder">AutoEncoder</TabsTrigger>
          </TabsList>
          
          <TabsContent value="transformer">
            <Card>
              <CardHeader>
                <CardTitle>Transformer Örneği</CardTitle>
                <CardDescription>
                  Basit bir metin sınıflandırma uygulaması
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global pooling
        return self.fc(x)

# Model oluşturma
model = TransformerClassifier(
    vocab_size=10000,
    d_model=256,
    nhead=8,
    num_layers=3,
    num_classes=2
)

# Model eğitimi
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
          
          <TabsContent value="gan">
            <Card>
              <CardHeader>
                <CardTitle>GAN Örneği</CardTitle>
                <CardDescription>
                  MNIST rakamları için GAN implementasyonu
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.model = nn.Sequential(
            # Giriş: latent_dim
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img.view(-1, 784))

# Model eğitimi
latent_dim = 100
generator = Generator(latent_dim)
discriminator = Discriminator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

for epoch in range(100):
    for real_imgs in train_loader:
        # Discriminator eğitimi
        d_optimizer.zero_grad()
        
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        
        real_loss = -torch.mean(torch.log(discriminator(real_imgs)))
        fake_loss = -torch.mean(torch.log(1 - discriminator(fake_imgs)))
        d_loss = real_loss + fake_loss
        
        d_loss.backward()
        d_optimizer.step()
        
        # Generator eğitimi
        g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        g_loss = -torch.mean(torch.log(discriminator(fake_imgs)))
        
        g_loss.backward()
        g_optimizer.step()`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="autoencoder">
            <Card>
              <CardHeader>
                <CardTitle>AutoEncoder Örneği</CardTitle>
                <CardDescription>
                  Görüntü sıkıştırma için VAE implementasyonu
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-secondary p-4 rounded-lg overflow-x-auto">
                  <code>{`class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# Model eğitimi
model = VAE(784, 400, 20)  # MNIST için
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction='sum'
    )
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

for epoch in range(100):
    for data in train_loader:
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        optimizer.step()`}</code>
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation */}
      <div className="mt-12 flex flex-col sm:flex-row justify-between gap-4">
        <Button asChild variant="outline" className="gap-2">
          <Link href="/topics/python/derin-ogrenme/tekrarlayan-sinir-aglari">
            <ArrowLeft className="h-4 w-4" />
            Önceki Konu: Tekrarlayan Sinir Ağları
          </Link>
        </Button>
        
        <Button asChild variant="default" className="gap-2">
          <Link href="/topics/python/derin-ogrenme/pratik-uygulamalar">
            Sonraki Konu: Pratik Uygulamalar
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