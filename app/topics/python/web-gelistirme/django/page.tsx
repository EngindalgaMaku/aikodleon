import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Django ile Web Geliştirme | Python Web Geliştirme | Kodleon',
  description: 'Django framework ile modern web uygulamaları geliştirmeyi öğrenin. MVT mimarisi, ORM, admin paneli, authentication ve daha fazlası.',
};

const content = `
# Django ile Web Geliştirme

Django, Python'da web uygulamaları geliştirmek için kullanılan güçlü ve tam donanımlı bir framework'tür. Bu bölümde, Django'nun temel özelliklerini ve ileri düzey konuları öğreneceğiz.

## Proje Yapısı ve Ayarlar

\`\`\`python
# settings.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-gizli-anahtar-123'

DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'blog.apps.BlogConfig',  # Blog uygulamamız
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'mysite.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Authentication ayarları
LOGIN_REDIRECT_URL = 'anasayfa'
LOGOUT_REDIRECT_URL = 'anasayfa'
\`\`\`

## Models (Veritabanı Modelleri)

\`\`\`python
# blog/models.py
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils.text import slugify

class Kategori(models.Model):
    ad = models.CharField(max_length=100)
    slug = models.SlugField(unique=True, blank=True)
    aciklama = models.TextField(blank=True)
    olusturma_tarihi = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.ad)
        super().save(*args, **kwargs)
    
    class Meta:
        verbose_name = 'Kategori'
        verbose_name_plural = 'Kategoriler'
        ordering = ['ad']
    
    def __str__(self):
        return self.ad
    
    def get_absolute_url(self):
        return reverse('kategori-detay', kwargs={'slug': self.slug})

class Gonderi(models.Model):
    DURUM_SECENEKLERI = [
        ('taslak', 'Taslak'),
        ('yayinda', 'Yayında'),
        ('arsivlendi', 'Arşivlendi'),
    ]
    
    baslik = models.CharField(max_length=200)
    slug = models.SlugField(unique=True, blank=True)
    icerik = models.TextField()
    ozet = models.TextField(blank=True)
    yazar = models.ForeignKey(User, on_delete=models.CASCADE)
    kategoriler = models.ManyToManyField(Kategori, related_name='gonderiler')
    durum = models.CharField(max_length=20, choices=DURUM_SECENEKLERI, default='taslak')
    olusturma_tarihi = models.DateTimeField(auto_now_add=True)
    guncelleme_tarihi = models.DateTimeField(auto_now=True)
    yayin_tarihi = models.DateTimeField(null=True, blank=True)
    kapak_resmi = models.ImageField(upload_to='gonderiler/%Y/%m/', blank=True)
    
    class Meta:
        verbose_name = 'Gönderi'
        verbose_name_plural = 'Gönderiler'
        ordering = ['-yayin_tarihi', '-olusturma_tarihi']
    
    def __str__(self):
        return self.baslik
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.baslik)
        if not self.ozet and self.icerik:
            self.ozet = self.icerik[:300]
        super().save(*args, **kwargs)
    
    def get_absolute_url(self):
        return reverse('gonderi-detay', kwargs={'slug': self.slug})
    
    @property
    def yorum_sayisi(self):
        return self.yorumlar.count()

class Yorum(models.Model):
    gonderi = models.ForeignKey(Gonderi, on_delete=models.CASCADE, related_name='yorumlar')
    yazar = models.ForeignKey(User, on_delete=models.CASCADE)
    icerik = models.TextField()
    olusturma_tarihi = models.DateTimeField(auto_now_add=True)
    aktif = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = 'Yorum'
        verbose_name_plural = 'Yorumlar'
        ordering = ['-olusturma_tarihi']
    
    def __str__(self):
        return f'{self.yazar.username} - {self.gonderi.baslik}'
\`\`\`

## Views (Görünümler)

\`\`\`python
# blog/views.py
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.db.models import Q
from django.contrib import messages
from .models import Gonderi, Kategori, Yorum
from .forms import GonderiForm, YorumForm

class GonderiListesi(ListView):
    model = Gonderi
    template_name = 'blog/gonderi_listesi.html'
    context_object_name = 'gonderiler'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Gonderi.objects.filter(durum='yayinda')
        kategori_slug = self.kwargs.get('kategori_slug')
        arama = self.request.GET.get('arama')
        
        if kategori_slug:
            queryset = queryset.filter(kategoriler__slug=kategori_slug)
        
        if arama:
            queryset = queryset.filter(
                Q(baslik__icontains=arama) |
                Q(icerik__icontains=arama) |
                Q(ozet__icontains=arama)
            )
        
        return queryset.select_related('yazar').prefetch_related('kategoriler')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['kategoriler'] = Kategori.objects.all()
        return context

class GonderiDetay(DetailView):
    model = Gonderi
    template_name = 'blog/gonderi_detay.html'
    context_object_name = 'gonderi'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['yorum_formu'] = YorumForm()
        context['yorumlar'] = self.object.yorumlar.filter(aktif=True)
        return context
    
    def post(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
            
        gonderi = self.get_object()
        form = YorumForm(request.POST)
        
        if form.is_valid():
            yorum = form.save(commit=False)
            yorum.gonderi = gonderi
            yorum.yazar = request.user
            yorum.save()
            messages.success(request, 'Yorumunuz başarıyla eklendi.')
            return redirect(gonderi.get_absolute_url())
        
        return self.get(request, *args, **kwargs)

class GonderiOlustur(LoginRequiredMixin, CreateView):
    model = Gonderi
    form_class = GonderiForm
    template_name = 'blog/gonderi_form.html'
    
    def form_valid(self, form):
        form.instance.yazar = self.request.user
        messages.success(self.request, 'Gönderi başarıyla oluşturuldu.')
        return super().form_valid(form)

class GonderiGuncelle(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Gonderi
    form_class = GonderiForm
    template_name = 'blog/gonderi_form.html'
    
    def test_func(self):
        gonderi = self.get_object()
        return self.request.user == gonderi.yazar
    
    def form_valid(self, form):
        messages.success(self.request, 'Gönderi başarıyla güncellendi.')
        return super().form_valid(form)

class GonderiSil(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Gonderi
    success_url = reverse_lazy('gonderi-listesi')
    
    def test_func(self):
        gonderi = self.get_object()
        return self.request.user == gonderi.yazar
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Gönderi başarıyla silindi.')
        return super().delete(request, *args, **kwargs)
\`\`\`

## Forms (Formlar)

\`\`\`python
# blog/forms.py
from django import forms
from .models import Gonderi, Yorum

class GonderiForm(forms.ModelForm):
    class Meta:
        model = Gonderi
        fields = ['baslik', 'icerik', 'ozet', 'kategoriler', 'durum', 'kapak_resmi']
        widgets = {
            'baslik': forms.TextInput(attrs={'class': 'form-control'}),
            'icerik': forms.Textarea(attrs={'class': 'form-control', 'rows': 10}),
            'ozet': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'kategoriler': forms.SelectMultiple(attrs={'class': 'form-control'}),
            'durum': forms.Select(attrs={'class': 'form-control'}),
        }

class YorumForm(forms.ModelForm):
    class Meta:
        model = Yorum
        fields = ['icerik']
        widgets = {
            'icerik': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Yorumunuzu buraya yazın...'
            })
        }
\`\`\`

## Admin Paneli

\`\`\`python
# blog/admin.py
from django.contrib import admin
from .models import Kategori, Gonderi, Yorum

@admin.register(Kategori)
class KategoriAdmin(admin.ModelAdmin):
    list_display = ['ad', 'slug', 'olusturma_tarihi']
    list_filter = ['olusturma_tarihi']
    search_fields = ['ad', 'aciklama']
    prepopulated_fields = {'slug': ('ad',)}

@admin.register(Gonderi)
class GonderiAdmin(admin.ModelAdmin):
    list_display = ['baslik', 'yazar', 'durum', 'yayin_tarihi', 'olusturma_tarihi']
    list_filter = ['durum', 'yayin_tarihi', 'kategoriler']
    search_fields = ['baslik', 'icerik', 'ozet']
    prepopulated_fields = {'slug': ('baslik',)}
    raw_id_fields = ['yazar']
    date_hierarchy = 'yayin_tarihi'
    filter_horizontal = ['kategoriler']
    
    def save_model(self, request, obj, form, change):
        if not obj.yazar_id:
            obj.yazar = request.user
        super().save_model(request, obj, form, change)

@admin.register(Yorum)
class YorumAdmin(admin.ModelAdmin):
    list_display = ['yazar', 'gonderi', 'olusturma_tarihi', 'aktif']
    list_filter = ['aktif', 'olusturma_tarihi']
    search_fields = ['yazar__username', 'icerik', 'gonderi__baslik']
    actions = ['yorum_aktif_yap', 'yorum_pasif_yap']
    
    def yorum_aktif_yap(self, request, queryset):
        queryset.update(aktif=True)
    yorum_aktif_yap.short_description = 'Seçili yorumları aktif yap'
    
    def yorum_pasif_yap(self, request, queryset):
        queryset.update(aktif=False)
    yorum_pasif_yap.short_description = 'Seçili yorumları pasif yap'
\`\`\`

## URL Yapılandırması

\`\`\`python
# blog/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.GonderiListesi.as_view(), name='gonderi-listesi'),
    path('kategori/<slug:kategori_slug>/', views.GonderiListesi.as_view(), name='kategori-detay'),
    path('gonderi/<slug:slug>/', views.GonderiDetay.as_view(), name='gonderi-detay'),
    path('gonderi/yeni/', views.GonderiOlustur.as_view(), name='gonderi-olustur'),
    path('gonderi/<slug:slug>/duzenle/', views.GonderiGuncelle.as_view(), name='gonderi-guncelle'),
    path('gonderi/<slug:slug>/sil/', views.GonderiSil.as_view(), name='gonderi-sil'),
]
\`\`\`

## Alıştırmalar

1. **Blog Uygulaması Geliştirme**
   - Kullanıcı profil sayfaları ekleyin
   - Gönderi etiketleme sistemi oluşturun
   - Sosyal medya paylaşım özellikleri ekleyin
   - Gönderi arşivleme sistemi geliştirin

2. **E-Ticaret Özellikleri**
   - Ürün kataloğu oluşturun
   - Alışveriş sepeti sistemi ekleyin
   - Ödeme entegrasyonu yapın
   - Sipariş takip sistemi geliştirin

3. **API Geliştirme**
   - Django REST framework kullanın
   - API authentication ekleyin
   - Swagger/OpenAPI dokümantasyonu oluşturun
   - API versiyonlama yapın

## Sonraki Adımlar

1. [FastAPI ile Modern API Geliştirme](/topics/python/web-gelistirme/fastapi)
2. [Web Deployment ve DevOps](/topics/python/web-gelistirme/deployment)
3. [Frontend Entegrasyonu](/topics/python/web-gelistirme/frontend)

## Faydalı Kaynaklar

- [Django Resmi Dokümantasyonu](https://docs.djangoproject.com/)
- [Django REST Framework Dokümantasyonu](https://www.django-rest-framework.org/)
- [Django Girls Tutorial](https://tutorial.djangogirls.org/tr/)
- [Two Scoops of Django](https://www.feldroy.com/books/two-scoops-of-django-3-x)
`;

const learningPath = [
  {
    title: '1. Django Temelleri',
    description: 'Django\'nun temel kavramlarını ve MVT mimarisini öğrenin.',
    topics: [
      'Proje yapısı ve ayarlar',
      'URL yapılandırması',
      'Views ve Templates',
      'Forms ve Validation',
      'Static ve Media dosyaları',
    ],
    icon: '🌱',
    href: '/topics/python/web-gelistirme/django/temeller'
  },
  {
    title: '2. Models ve ORM',
    description: 'Django ORM ile veritabanı işlemlerini öğrenin.',
    topics: [
      'Model tanımlama',
      'Querysets ve Managers',
      'İlişkisel veritabanı',
      'Migrations',
      'Model inheritance',
    ],
    icon: '💾',
    href: '/topics/python/web-gelistirme/django/models'
  },
  {
    title: '3. Admin ve Authentication',
    description: 'Django admin paneli ve kullanıcı yönetimini öğrenin.',
    topics: [
      'Admin panel özelleştirme',
      'User modeli ve authentication',
      'Permissions ve Groups',
      'Custom user modeli',
      'Password management',
    ],
    icon: '👤',
    href: '/topics/python/web-gelistirme/django/admin'
  },
  {
    title: '4. Advanced Django',
    description: 'İleri düzey Django özelliklerini ve best practice\'leri öğrenin.',
    topics: [
      'Class-based views',
      'Middleware',
      'Signals',
      'Caching',
      'Testing',
    ],
    icon: '🚀',
    href: '/topics/python/web-gelistirme/django/advanced'
  }
];

export default function DjangoPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/web-gelistirme" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Web Geliştirme
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