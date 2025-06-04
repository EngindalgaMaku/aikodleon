import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'FastAPI ile Modern API Geliştirme | Python Web Geliştirme | Kodleon',
  description: 'FastAPI framework ile modern, hızlı ve asenkron API\'lar geliştirmeyi öğrenin. Type hints, otomatik API dokümantasyonu, dependency injection ve daha fazlası.',
};

const content = `
# FastAPI ile Modern API Geliştirme

FastAPI, modern Python web uygulamaları geliştirmek için kullanılan hızlı, asenkron ve type-safe bir framework'tür. Bu bölümde, FastAPI'nin temel özelliklerini ve ileri düzey konuları öğreneceğiz.

## Temel Yapı ve Ayarlar

\`\`\`python
# main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import uvicorn

# API uygulaması oluşturma
app = FastAPI(
    title="Blog API",
    description="Modern blog API'si",
    version="1.0.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Veritabanı bağlantısı
from database import SessionLocal, engine
import models

models.Base.metadata.create_all(bind=engine)

# Dependency injection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    duration = datetime.now() - start_time
    
    print(f"{request.method} {request.url} {response.status_code} {duration}")
    return response
\`\`\`

## Modeller ve Şemalar

\`\`\`python
# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class Gonderi(Base):
    __tablename__ = "gonderiler"
    
    id = Column(Integer, primary_key=True, index=True)
    baslik = Column(String(200), nullable=False)
    icerik = Column(Text, nullable=False)
    ozet = Column(Text)
    yazar_id = Column(Integer, ForeignKey("kullanicilar.id"))
    olusturma_tarihi = Column(DateTime, default=datetime.utcnow)
    guncelleme_tarihi = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    aktif = Column(Boolean, default=True)
    
    yazar = relationship("Kullanici", back_populates="gonderiler")
    yorumlar = relationship("Yorum", back_populates="gonderi", cascade="all, delete-orphan")

class Kullanici(Base):
    __tablename__ = "kullanicilar"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(100))
    ad = Column(String(50))
    soyad = Column(String(50))
    aktif = Column(Boolean, default=True)
    
    gonderiler = relationship("Gonderi", back_populates="yazar")
    yorumlar = relationship("Yorum", back_populates="yazar")

class Yorum(Base):
    __tablename__ = "yorumlar"
    
    id = Column(Integer, primary_key=True, index=True)
    icerik = Column(Text, nullable=False)
    gonderi_id = Column(Integer, ForeignKey("gonderiler.id"))
    yazar_id = Column(Integer, ForeignKey("kullanicilar.id"))
    olusturma_tarihi = Column(DateTime, default=datetime.utcnow)
    aktif = Column(Boolean, default=True)
    
    gonderi = relationship("Gonderi", back_populates="yorumlar")
    yazar = relationship("Kullanici", back_populates="yorumlar")

# schemas.py
from pydantic import BaseModel, EmailStr, constr
from typing import Optional, List
from datetime import datetime

class YorumBase(BaseModel):
    icerik: str

class YorumCreate(YorumBase):
    pass

class Yorum(YorumBase):
    id: int
    gonderi_id: int
    yazar_id: int
    olusturma_tarihi: datetime
    aktif: bool
    
    class Config:
        from_attributes = True

class GonderiBase(BaseModel):
    baslik: constr(min_length=3, max_length=200)
    icerik: str
    ozet: Optional[str] = None

class GonderiCreate(GonderiBase):
    pass

class Gonderi(GonderiBase):
    id: int
    yazar_id: int
    olusturma_tarihi: datetime
    guncelleme_tarihi: datetime
    aktif: bool
    yorumlar: List[Yorum] = []
    
    class Config:
        from_attributes = True

class KullaniciBase(BaseModel):
    email: EmailStr
    username: constr(min_length=3, max_length=50)
    ad: Optional[str] = None
    soyad: Optional[str] = None

class KullaniciCreate(KullaniciBase):
    password: constr(min_length=8)

class Kullanici(KullaniciBase):
    id: int
    aktif: bool
    gonderiler: List[Gonderi] = []
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
\`\`\`

## Güvenlik ve Kimlik Doğrulama

\`\`\`python
# security.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

# Güvenlik ayarları
SECRET_KEY = "gizli-anahtar-123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Geçersiz kimlik bilgileri",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Kullanici = Depends(get_current_user)):
    if not current_user.aktif:
        raise HTTPException(status_code=400, detail="Pasif kullanıcı")
    return current_user
\`\`\`

## API Endpoint'leri

\`\`\`python
# routers/blog.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from . import crud, schemas
from .database import get_db
from .security import get_current_active_user

router = APIRouter(
    prefix="/api/v1",
    tags=["blog"]
)

@router.post("/gonderiler/", response_model=schemas.Gonderi)
async def gonderi_olustur(
    gonderi: schemas.GonderiCreate,
    db: Session = Depends(get_db),
    current_user: schemas.Kullanici = Depends(get_current_active_user)
):
    """
    Yeni bir blog gönderisi oluştur.
    
    - **baslik**: Gönderi başlığı (3-200 karakter)
    - **icerik**: Gönderi içeriği
    - **ozet**: Gönderi özeti (opsiyonel)
    """
    return crud.create_gonderi(db=db, gonderi=gonderi, user_id=current_user.id)

@router.get("/gonderiler/", response_model=List[schemas.Gonderi])
async def gonderileri_listele(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Blog gönderilerini listele.
    
    - **skip**: Atlanacak gönderi sayısı (sayfalama için)
    - **limit**: Sayfa başına gönderi sayısı
    """
    gonderiler = crud.get_gonderiler(db, skip=skip, limit=limit)
    return gonderiler

@router.get("/gonderiler/{gonderi_id}", response_model=schemas.Gonderi)
async def gonderi_detay(gonderi_id: int, db: Session = Depends(get_db)):
    """
    Belirli bir blog gönderisinin detaylarını getir.
    
    - **gonderi_id**: Gönderi ID
    """
    gonderi = crud.get_gonderi(db, gonderi_id=gonderi_id)
    if gonderi is None:
        raise HTTPException(status_code=404, detail="Gönderi bulunamadı")
    return gonderi

@router.put("/gonderiler/{gonderi_id}", response_model=schemas.Gonderi)
async def gonderi_guncelle(
    gonderi_id: int,
    gonderi: schemas.GonderiCreate,
    db: Session = Depends(get_db),
    current_user: schemas.Kullanici = Depends(get_current_active_user)
):
    """
    Bir blog gönderisini güncelle.
    
    - **gonderi_id**: Güncellenecek gönderi ID
    - **gonderi**: Yeni gönderi bilgileri
    """
    db_gonderi = crud.get_gonderi(db, gonderi_id=gonderi_id)
    if db_gonderi is None:
        raise HTTPException(status_code=404, detail="Gönderi bulunamadı")
    if db_gonderi.yazar_id != current_user.id:
        raise HTTPException(status_code=403, detail="Bu işlem için yetkiniz yok")
    return crud.update_gonderi(db=db, gonderi_id=gonderi_id, gonderi=gonderi)

@router.delete("/gonderiler/{gonderi_id}")
async def gonderi_sil(
    gonderi_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.Kullanici = Depends(get_current_active_user)
):
    """
    Bir blog gönderisini sil.
    
    - **gonderi_id**: Silinecek gönderi ID
    """
    db_gonderi = crud.get_gonderi(db, gonderi_id=gonderi_id)
    if db_gonderi is None:
        raise HTTPException(status_code=404, detail="Gönderi bulunamadı")
    if db_gonderi.yazar_id != current_user.id:
        raise HTTPException(status_code=403, detail="Bu işlem için yetkiniz yok")
    crud.delete_gonderi(db=db, gonderi_id=gonderi_id)
    return {"message": "Gönderi başarıyla silindi"}
\`\`\`

## Veritabanı İşlemleri

\`\`\`python
# crud.py
from sqlalchemy.orm import Session
from . import models, schemas
from fastapi import HTTPException
from .security import get_password_hash

def get_user(db: Session, user_id: int):
    return db.query(models.Kullanici).filter(models.Kullanici.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.Kullanici).filter(models.Kullanici.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Kullanici).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.KullaniciCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.Kullanici(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        ad=user.ad,
        soyad=user.soyad
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_gonderiler(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Gonderi).offset(skip).limit(limit).all()

def create_gonderi(db: Session, gonderi: schemas.GonderiCreate, user_id: int):
    db_gonderi = models.Gonderi(**gonderi.dict(), yazar_id=user_id)
    db.add(db_gonderi)
    db.commit()
    db.refresh(db_gonderi)
    return db_gonderi

def update_gonderi(db: Session, gonderi_id: int, gonderi: schemas.GonderiCreate):
    db_gonderi = db.query(models.Gonderi).filter(models.Gonderi.id == gonderi_id)
    db_gonderi.update(gonderi.dict())
    db.commit()
    return db_gonderi.first()

def delete_gonderi(db: Session, gonderi_id: int):
    db_gonderi = db.query(models.Gonderi).filter(models.Gonderi.id == gonderi_id)
    db_gonderi.delete()
    db.commit()
\`\`\`

## Alıştırmalar

1. **API Geliştirme**
   - Kullanıcı yetkilendirme sistemi ekleyin
   - Rate limiting uygulayın
   - Dosya yükleme endpoint'i oluşturun
   - API versiyonlama yapın

2. **Veritabanı İşlemleri**
   - Asenkron veritabanı bağlantısı kurun
   - Migration sistemi ekleyin
   - Veritabanı indeksleme yapın
   - Caching mekanizması ekleyin

3. **Test ve Dokümantasyon**
   - Unit testler yazın
   - Integration testler ekleyin
   - API dokümantasyonunu özelleştirin
   - Postman koleksiyonu oluşturun

## Sonraki Adımlar

1. [Web Deployment ve DevOps](/topics/python/web-gelistirme/deployment)
2. [Frontend Entegrasyonu](/topics/python/web-gelistirme/frontend)
3. [Mikroservis Mimarisi](/topics/python/web-gelistirme/microservices)

## Faydalı Kaynaklar

- [FastAPI Resmi Dokümantasyonu](https://fastapi.tiangolo.com/)
- [SQLAlchemy Dokümantasyonu](https://docs.sqlalchemy.org/)
- [Pydantic Dokümantasyonu](https://pydantic-docs.helpmanual.io/)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
`;

const learningPath = [
  {
    title: '1. FastAPI Temelleri',
    description: 'FastAPI\'nin temel kavramlarını ve yapısını öğrenin.',
    topics: [
      'Async/await programlama',
      'Path ve Query parametreleri',
      'Request ve Response modelleri',
      'Dependency Injection',
      'Middleware ve CORS',
    ],
    icon: '🚀',
    href: '/topics/python/web-gelistirme/fastapi/temeller'
  },
  {
    title: '2. Veritabanı ve ORM',
    description: 'SQLAlchemy ile veritabanı işlemlerini öğrenin.',
    topics: [
      'Async SQLAlchemy',
      'Model tanımlama',
      'Migration yönetimi',
      'İlişkisel veritabanı',
      'Query optimizasyonu',
    ],
    icon: '💾',
    href: '/topics/python/web-gelistirme/fastapi/veritabani'
  },
  {
    title: '3. Güvenlik ve Auth',
    description: 'API güvenliği ve kimlik doğrulama sistemlerini öğrenin.',
    topics: [
      'JWT authentication',
      'OAuth2 ve OpenID Connect',
      'Role-based access control',
      'Rate limiting',
      'API keys',
    ],
    icon: '🔐',
    href: '/topics/python/web-gelistirme/fastapi/guvenlik'
  },
  {
    title: '4. Testing ve Deployment',
    description: 'API testleri ve deployment süreçlerini öğrenin.',
    topics: [
      'Unit ve integration testler',
      'Docker containerization',
      'CI/CD pipeline',
      'Performance monitoring',
      'Load balancing',
    ],
    icon: '🔧',
    href: '/topics/python/web-gelistirme/fastapi/testing'
  }
];

export default function FastAPIPage() {
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