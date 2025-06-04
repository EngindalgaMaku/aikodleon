import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Web Geliştirme | Python | Kodleon',
  description: 'Python ile modern web geliştirme. Flask, Django ve FastAPI kullanarak web uygulamaları geliştirme.',
};

const content = `
# Python ile Web Geliştirme

Bu bölümde, Python'un popüler web framework'lerini ve modern web geliştirme tekniklerini öğreneceğiz.

## Flask ile Web Geliştirme

\`\`\`python
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dataclasses import dataclass
from datetime import datetime
import logging

# Uygulama konfigürasyonu
@dataclass
class UygulamaKonfigurasyonu:
    veritabani_url: str
    gizli_anahtar: str
    debug: bool = False
    port: int = 5000
    host: str = 'localhost'
    
class WebUygulamasi:
    def __init__(self, konfigurasyon: UygulamaKonfigurasyonu):
        self.konfigurasyon = konfigurasyon
        
        # Flask uygulamasını oluştur
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = konfigurasyon.gizli_anahtar
        self.app.config['SQLALCHEMY_DATABASE_URI'] = konfigurasyon.veritabani_url
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        # Veritabanı
        self.db = SQLAlchemy(self.app)
        self.migrate = Migrate(self.app, self.db)
        
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Route'ları kaydet
        self._route_kaydet()
        
    def _route_kaydet(self):
        """API endpoint'lerini tanımlar"""
        
        @self.app.route('/')
        def anasayfa():
            return render_template('index.html')
            
        @self.app.route('/api/kullanicilar', methods=['GET'])
        def kullanicilari_getir():
            try:
                kullanicilar = Kullanici.query.all()
                return jsonify([{
                    'id': k.id,
                    'ad': k.ad,
                    'email': k.email,
                    'kayit_tarihi': k.kayit_tarihi.isoformat()
                } for k in kullanicilar])
            except Exception as e:
                self.logger.error(f"Kullanıcı listesi alınamadı: {str(e)}")
                return jsonify({'hata': 'Kullanıcılar getirilemedi'}), 500
                
        @self.app.route('/api/kullanicilar', methods=['POST'])
        def kullanici_ekle():
            try:
                veri = request.get_json()
                
                yeni_kullanici = Kullanici(
                    ad=veri['ad'],
                    email=veri['email'],
                    sifre=veri['sifre']
                )
                
                self.db.session.add(yeni_kullanici)
                self.db.session.commit()
                
                return jsonify({
                    'mesaj': 'Kullanıcı başarıyla eklendi',
                    'kullanici_id': yeni_kullanici.id
                }), 201
                
            except KeyError as e:
                return jsonify({'hata': f'Eksik alan: {str(e)}'}), 400
            except Exception as e:
                self.logger.error(f"Kullanıcı eklenemedi: {str(e)}")
                self.db.session.rollback()
                return jsonify({'hata': 'Kullanıcı eklenemedi'}), 500
                
        @self.app.route('/api/kullanicilar/<int:kullanici_id>', methods=['PUT'])
        def kullanici_guncelle(kullanici_id: int):
            try:
                kullanici = Kullanici.query.get_or_404(kullanici_id)
                veri = request.get_json()
                
                if 'ad' in veri:
                    kullanici.ad = veri['ad']
                if 'email' in veri:
                    kullanici.email = veri['email']
                    
                self.db.session.commit()
                
                return jsonify({
                    'mesaj': 'Kullanıcı güncellendi',
                    'kullanici': {
                        'id': kullanici.id,
                        'ad': kullanici.ad,
                        'email': kullanici.email
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Kullanıcı güncellenemedi: {str(e)}")
                self.db.session.rollback()
                return jsonify({'hata': 'Kullanıcı güncellenemedi'}), 500
                
        @self.app.route('/api/kullanicilar/<int:kullanici_id>', methods=['DELETE'])
        def kullanici_sil(kullanici_id: int):
            try:
                kullanici = Kullanici.query.get_or_404(kullanici_id)
                self.db.session.delete(kullanici)
                self.db.session.commit()
                
                return jsonify({
                    'mesaj': 'Kullanıcı silindi',
                    'kullanici_id': kullanici_id
                })
                
            except Exception as e:
                self.logger.error(f"Kullanıcı silinemedi: {str(e)}")
                self.db.session.rollback()
                return jsonify({'hata': 'Kullanıcı silinemedi'}), 500
                
    def calistir(self):
        """Uygulamayı başlatır"""
        self.app.run(
            host=self.konfigurasyon.host,
            port=self.konfigurasyon.port,
            debug=self.konfigurasyon.debug
        )

# Veritabanı modelleri
class Kullanici(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ad = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    sifre = db.Column(db.String(100), nullable=False)
    kayit_tarihi = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Kullanici {self.email}>'

# Kullanım örneği
if __name__ == '__main__':
    # Konfigürasyon
    konfig = UygulamaKonfigurasyonu(
        veritabani_url='sqlite:///uygulama.db',
        gizli_anahtar='gizli-anahtar-123',
        debug=True
    )
    
    # Uygulamayı oluştur ve çalıştır
    uygulama = WebUygulamasi(konfig)
    uygulama.calistir()
\`\`\`

## Django ile Web Geliştirme

\`\`\`python
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.urls import path
from django.views.generic import ListView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

# Modeller
class Kullanici(AbstractUser):
    bio = models.TextField(max_length=500, blank=True)
    dogum_tarihi = models.DateField(null=True, blank=True)
    profil_resmi = models.ImageField(upload_to='profil_resimleri/', null=True, blank=True)
    
    class Meta:
        verbose_name = 'Kullanıcı'
        verbose_name_plural = 'Kullanıcılar'
        
class Gonderi(models.Model):
    baslik = models.CharField(max_length=200)
    icerik = models.TextField()
    yazar = models.ForeignKey(Kullanici, on_delete=models.CASCADE)
    olusturma_tarihi = models.DateTimeField(auto_now_add=True)
    guncelleme_tarihi = models.DateTimeField(auto_now=True)
    kategoriler = models.ManyToManyField('Kategori')
    
    class Meta:
        verbose_name = 'Gönderi'
        verbose_name_plural = 'Gönderiler'
        ordering = ['-olusturma_tarihi']
        
    def __str__(self):
        return self.baslik
        
class Kategori(models.Model):
    ad = models.CharField(max_length=100)
    aciklama = models.TextField(blank=True)
    
    class Meta:
        verbose_name = 'Kategori'
        verbose_name_plural = 'Kategoriler'
        
    def __str__(self):
        return self.ad

# Viewsets
class GonderiViewSet(viewsets.ModelViewSet):
    queryset = Gonderi.objects.all()
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    
    def get_queryset(self):
        queryset = Gonderi.objects.all()
        kategori = self.request.query_params.get('kategori', None)
        
        if kategori:
            queryset = queryset.filter(kategoriler__ad=kategori)
            
        return queryset
        
    @action(detail=True, methods=['post'])
    def begen(self, request, pk=None):
        gonderi = self.get_object()
        # Beğeni işlemleri...
        return Response({'status': 'beğenildi'})
        
# Views
class GonderiListesi(LoginRequiredMixin, ListView):
    model = Gonderi
    template_name = 'blog/gonderi_listesi.html'
    context_object_name = 'gonderiler'
    paginate_by = 10
    
    def get_queryset(self):
        return Gonderi.objects.select_related('yazar').prefetch_related('kategoriler')
        
class GonderiOlustur(LoginRequiredMixin, CreateView):
    model = Gonderi
    template_name = 'blog/gonderi_form.html'
    fields = ['baslik', 'icerik', 'kategoriler']
    
    def form_valid(self, form):
        form.instance.yazar = self.request.user
        return super().form_valid(form)
        
# URLs
urlpatterns = [
    path('gonderiler/', GonderiListesi.as_view(), name='gonderi-listesi'),
    path('gonderi/yeni/', GonderiOlustur.as_view(), name='gonderi-olustur'),
]

# Forms
from django import forms

class GonderiFormu(forms.ModelForm):
    class Meta:
        model = Gonderi
        fields = ['baslik', 'icerik', 'kategoriler']
        widgets = {
            'baslik': forms.TextInput(attrs={'class': 'form-control'}),
            'icerik': forms.Textarea(attrs={'class': 'form-control'}),
            'kategoriler': forms.SelectMultiple(attrs={'class': 'form-control'})
        }
\`\`\`

## FastAPI ile Web Geliştirme

\`\`\`python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import logging
from dataclasses import dataclass

# Konfigürasyon
@dataclass
class APIKonfigurasyonu:
    veritabani_url: str
    gizli_anahtar: str
    token_suresi: int = 30  # dakika
    algoritma: str = "HS256"

# Pydantic modelleri
class KullaniciOlustur(BaseModel):
    email: EmailStr
    sifre: str
    ad: str
    soyad: str

class Kullanici(BaseModel):
    id: int
    email: EmailStr
    ad: str
    soyad: str
    aktif: bool

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

# API uygulaması
class APIUygulamasi:
    def __init__(self, konfigurasyon: APIKonfigurasyonu):
        self.konfigurasyon = konfigurasyon
        self.app = FastAPI(title="FastAPI Örnek", version="1.0.0")
        
        # Güvenlik
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Route'ları kaydet
        self._route_kaydet()
        
    def _route_kaydet(self):
        @self.app.post("/token", response_model=Token)
        async def token_olustur(form_data: OAuth2PasswordRequestForm = Depends()):
            kullanici = self._kullaniciyi_dogrula(form_data.username, form_data.password)
            if not kullanici:
                raise HTTPException(
                    status_code=400,
                    detail="Incorrect username or password"
                )
                
            token = self._access_token_olustur(
                data={"sub": kullanici.email}
            )
            
            return {"access_token": token, "token_type": "bearer"}
            
        @self.app.post("/kullanicilar/", response_model=Kullanici)
        async def kullanici_olustur(kullanici: KullaniciOlustur):
            return self._yeni_kullanici_olustur(kullanici)
            
        @self.app.get("/kullanicilar/me", response_model=Kullanici)
        async def mevcut_kullanici(current_user: Kullanici = Depends(self._mevcut_kullanici)):
            return current_user
            
    def _sifre_dogrula(self, duz_sifre: str, hash_sifre: str) -> bool:
        return self.pwd_context.verify(duz_sifre, hash_sifre)
        
    def _sifre_hashle(self, sifre: str) -> str:
        return self.pwd_context.hash(sifre)
        
    def _access_token_olustur(self, data: dict) -> str:
        to_encode = data.copy()
        sure = datetime.utcnow() + timedelta(minutes=self.konfigurasyon.token_suresi)
        to_encode.update({"exp": sure})
        
        return jwt.encode(
            to_encode,
            self.konfigurasyon.gizli_anahtar,
            algorithm=self.konfigurasyon.algoritma
        )
        
    async def _mevcut_kullanici(self, token: str = Depends(oauth2_scheme)) -> Kullanici:
        credentials_exception = HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(
                token,
                self.konfigurasyon.gizli_anahtar,
                algorithms=[self.konfigurasyon.algoritma]
            )
            email: str = payload.get("sub")
            if email is None:
                raise credentials_exception
        except jwt.JWTError:
            raise credentials_exception
            
        kullanici = self._kullaniciyi_bul(email)
        if kullanici is None:
            raise credentials_exception
            
        return kullanici

# Kullanım örneği
if __name__ == "__main__":
    # Konfigürasyon
    konfig = APIKonfigurasyonu(
        veritabani_url="postgresql://user:password@localhost/dbname",
        gizli_anahtar="your-secret-key"
    )
    
    # API uygulamasını oluştur
    api = APIUygulamasi(konfig)
    
    # Uvicorn ile çalıştır
    import uvicorn
    uvicorn.run(api.app, host="0.0.0.0", port=8000)
\`\`\`

## Alıştırmalar

1. **Flask ile Blog Uygulaması**
   - Kullanıcı yönetimi ekleyin
   - Gönderi CRUD işlemlerini gerçekleştirin
   - Şablon sistemini kullanın
   - Veritabanı ilişkilerini kurun

2. **Django ile E-Ticaret Sitesi**
   - Ürün kataloğu oluşturun
   - Alışveriş sepeti sistemi ekleyin
   - Ödeme entegrasyonu yapın
   - Admin panelini özelleştirin

3. **FastAPI ile REST API**
   - JWT authentication ekleyin
   - API dokümantasyonu oluşturun
   - Rate limiting uygulayın
   - Asenkron veritabanı işlemleri yapın

## Sonraki Adımlar

1. [API Geliştirme](/topics/python/web-gelistirme/api)
2. [Frontend Entegrasyonu](/topics/python/web-gelistirme/frontend)
3. [Deployment ve DevOps](/topics/python/web-gelistirme/deployment)

## Faydalı Kaynaklar

- [Flask Dokümantasyonu](https://flask.palletsprojects.com/)
- [Django Dokümantasyonu](https://docs.djangoproject.com/)
- [FastAPI Dokümantasyonu](https://fastapi.tiangolo.com/)
- [Python Web Development Guide](https://www.fullstackpython.com/)
`;

const learningPath = [
  {
    title: '1. Flask ile Web Geliştirme',
    description: 'Hafif ve esnek bir web framework olan Flask ile web uygulamaları geliştirin.',
    topics: [
      'Temel Flask kavramları',
      'Routing ve views',
      'Şablonlar ve formlar',
      'Veritabanı entegrasyonu',
      'RESTful API geliştirme',
    ],
    icon: '🌐',
    href: '/topics/python/web-gelistirme/flask'
  },
  {
    title: '2. Django ile Web Geliştirme',
    description: 'Tam özellikli bir web framework olan Django ile kapsamlı web uygulamaları oluşturun.',
    topics: [
      'MVT mimarisi',
      'Admin paneli',
      'ORM ve migrations',
      'Authentication ve authorization',
      'Forms ve Class-based views',
    ],
    icon: '🎯',
    href: '/topics/python/web-gelistirme/django'
  },
  {
    title: '3. FastAPI ile Modern API Geliştirme',
    description: 'Modern ve hızlı API geliştirme framework\'ü FastAPI ile REST API\'lar oluşturun.',
    topics: [
      'Async/await yapıları',
      'Pydantic modelleri',
      'OpenAPI/Swagger',
      'Dependency injection',
      'WebSocket desteği',
    ],
    icon: '⚡',
    href: '/topics/python/web-gelistirme/fastapi'
  },
  {
    title: '4. Web Deployment ve DevOps',
    description: 'Web uygulamalarını deploy etme ve DevOps pratiklerini uygulama.',
    topics: [
      'Docker containerization',
      'CI/CD pipeline\'ları',
      'Cloud deployment',
      'Monitoring ve logging',
      'Performance optimizasyonu',
    ],
    icon: '🚀',
    href: '/topics/python/web-gelistirme/deployment'
  }
];

export default function WebGelistirmePage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Python
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