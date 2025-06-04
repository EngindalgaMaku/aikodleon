import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Flask ile Web Geliştirme | Python Web Geliştirme | Kodleon',
  description: 'Flask framework ile modern web uygulamaları geliştirmeyi öğrenin. Routing, templates, veritabanı entegrasyonu ve daha fazlası.',
};

const content = `
# Flask ile Web Geliştirme

Flask, Python'da web uygulamaları geliştirmek için kullanılan hafif ve esnek bir framework'tür. Bu bölümde, Flask ile web geliştirmenin temellerini ve ileri düzey konuları öğreneceğiz.

## Temel Flask Uygulaması

\`\`\`python
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import logging

# Uygulama oluşturma
app = Flask(__name__)
app.config['SECRET_KEY'] = 'gizli-anahtar-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Veritabanı
db = SQLAlchemy(app)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model tanımları
class Gonderi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    baslik = db.Column(db.String(200), nullable=False)
    icerik = db.Column(db.Text, nullable=False)
    yazar = db.Column(db.String(100))
    olusturma_tarihi = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Gonderi {self.baslik}>'
        
    def to_dict(self):
        return {
            'id': self.id,
            'baslik': self.baslik,
            'icerik': self.icerik,
            'yazar': self.yazar,
            'olusturma_tarihi': self.olusturma_tarihi.isoformat()
        }

# Route'lar
@app.route('/')
def anasayfa():
    gonderiler = Gonderi.query.order_by(Gonderi.olusturma_tarihi.desc()).all()
    return render_template('anasayfa.html', gonderiler=gonderiler)

@app.route('/gonderi/<int:gonderi_id>')
def gonderi_detay(gonderi_id):
    gonderi = Gonderi.query.get_or_404(gonderi_id)
    return render_template('gonderi_detay.html', gonderi=gonderi)

@app.route('/gonderi/yeni', methods=['GET', 'POST'])
def gonderi_olustur():
    if request.method == 'POST':
        try:
            yeni_gonderi = Gonderi(
                baslik=request.form['baslik'],
                icerik=request.form['icerik'],
                yazar=request.form['yazar']
            )
            db.session.add(yeni_gonderi)
            db.session.commit()
            logger.info(f"Yeni gönderi oluşturuldu: {yeni_gonderi.baslik}")
            return redirect(url_for('gonderi_detay', gonderi_id=yeni_gonderi.id))
        except Exception as e:
            logger.error(f"Gönderi oluşturma hatası: {str(e)}")
            db.session.rollback()
            flash('Gönderi oluşturulurken bir hata oluştu', 'error')
    
    return render_template('gonderi_form.html')

# API endpoints
@app.route('/api/gonderiler')
def api_gonderiler():
    try:
        gonderiler = Gonderi.query.all()
        return jsonify([gonderi.to_dict() for gonderi in gonderiler])
    except Exception as e:
        logger.error(f"API hatası: {str(e)}")
        return jsonify({'hata': 'Gönderiler alınamadı'}), 500

@app.route('/api/gonderi/<int:gonderi_id>', methods=['GET', 'PUT', 'DELETE'])
def api_gonderi_islemleri(gonderi_id):
    gonderi = Gonderi.query.get_or_404(gonderi_id)
    
    if request.method == 'GET':
        return jsonify(gonderi.to_dict())
        
    elif request.method == 'PUT':
        try:
            data = request.get_json()
            gonderi.baslik = data.get('baslik', gonderi.baslik)
            gonderi.icerik = data.get('icerik', gonderi.icerik)
            gonderi.yazar = data.get('yazar', gonderi.yazar)
            db.session.commit()
            return jsonify(gonderi.to_dict())
        except Exception as e:
            logger.error(f"Güncelleme hatası: {str(e)}")
            db.session.rollback()
            return jsonify({'hata': 'Gönderi güncellenemedi'}), 500
            
    elif request.method == 'DELETE':
        try:
            db.session.delete(gonderi)
            db.session.commit()
            return jsonify({'mesaj': 'Gönderi silindi'})
        except Exception as e:
            logger.error(f"Silme hatası: {str(e)}")
            db.session.rollback()
            return jsonify({'hata': 'Gönderi silinemedi'}), 500

# Hata yönetimi
@app.errorhandler(404)
def sayfa_bulunamadi(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def sunucu_hatasi(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
\`\`\`

## Şablonlar (Templates)

\`\`\`html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Flask Blog{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <a href="{{ url_for('anasayfa') }}" class="navbar-brand">Flask Blog</a>
            <ul class="navbar-nav">
                <li><a href="{{ url_for('anasayfa') }}">Anasayfa</a></li>
                <li><a href="{{ url_for('gonderi_olustur') }}">Yeni Gönderi</a></li>
            </ul>
        </div>
    </nav>

    <main class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {% block content %}{% endblock %}
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; {{ year }} Flask Blog. Tüm hakları saklıdır.</p>
        </div>
    </footer>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>

<!-- templates/anasayfa.html -->
{% extends "base.html" %}

{% block title %}Anasayfa | Flask Blog{% endblock %}

{% block content %}
<h1>Son Gönderiler</h1>

<div class="gonderiler">
    {% for gonderi in gonderiler %}
    <article class="gonderi-card">
        <h2>{{ gonderi.baslik }}</h2>
        <p class="gonderi-meta">
            {{ gonderi.yazar }} tarafından {{ gonderi.olusturma_tarihi.strftime('%d.%m.%Y') }}
        </p>
        <p class="gonderi-ozet">{{ gonderi.icerik[:200] }}...</p>
        <a href="{{ url_for('gonderi_detay', gonderi_id=gonderi.id) }}" class="btn">Devamını Oku</a>
    </article>
    {% else %}
    <p>Henüz gönderi bulunmuyor.</p>
    {% endfor %}
</div>
{% endblock %}

<!-- templates/gonderi_form.html -->
{% extends "base.html" %}

{% block title %}Yeni Gönderi | Flask Blog{% endblock %}

{% block content %}
<h1>Yeni Gönderi Oluştur</h1>

<form method="POST" class="gonderi-form">
    <div class="form-group">
        <label for="baslik">Başlık</label>
        <input type="text" id="baslik" name="baslik" required>
    </div>

    <div class="form-group">
        <label for="icerik">İçerik</label>
        <textarea id="icerik" name="icerik" rows="10" required></textarea>
    </div>

    <div class="form-group">
        <label for="yazar">Yazar</label>
        <input type="text" id="yazar" name="yazar" required>
    </div>

    <button type="submit" class="btn btn-primary">Gönderiyi Yayınla</button>
</form>
{% endblock %}
\`\`\`

## Statik Dosyalar

\`\`\`css
/* static/css/style.css */
:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --success-color: #28a745;
    --danger-color: #dc3545;
    --light-color: #f8f9fa;
    --dark-color: #343a40;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 15px;
}

.navbar {
    background-color: var(--dark-color);
    padding: 1rem 0;
    margin-bottom: 2rem;
}

.navbar-brand {
    color: white;
    font-size: 1.5rem;
    text-decoration: none;
}

.navbar-nav {
    display: flex;
    list-style: none;
    margin: 0;
    padding: 0;
}

.navbar-nav li a {
    color: white;
    text-decoration: none;
    padding: 0.5rem 1rem;
}

.gonderi-card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
    padding: 1.5rem;
}

.gonderi-meta {
    color: var(--secondary-color);
    font-size: 0.9rem;
}

.btn {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    text-decoration: none;
    cursor: pointer;
}

.btn-primary {
    background-color: var(--primary-color);
    color: white;
    border: none;
}

.form-group {
    margin-bottom: 1rem;
}

.form-group label {
    display: block;
    margin-bottom: 0.5rem;
}

.form-group input,
.form-group textarea {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.alert {
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 1rem;
}

.alert-error {
    background-color: var(--danger-color);
    color: white;
}

.footer {
    background-color: var(--light-color);
    padding: 2rem 0;
    margin-top: 3rem;
    text-align: center;
}
\`\`\`

## Veritabanı Şeması

\`\`\`python
from flask_migrate import Migrate
from datetime import datetime

# Veritabanı migrasyonları için
migrate = Migrate(app, db)

class Kategori(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ad = db.Column(db.String(100), unique=True, nullable=False)
    aciklama = db.Column(db.Text)
    gonderiler = db.relationship('Gonderi', secondary='gonderi_kategori', backref='kategoriler')

class Yorum(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    icerik = db.Column(db.Text, nullable=False)
    yazar = db.Column(db.String(100))
    olusturma_tarihi = db.Column(db.DateTime, default=datetime.utcnow)
    gonderi_id = db.Column(db.Integer, db.ForeignKey('gonderi.id'), nullable=False)
    gonderi = db.relationship('Gonderi', backref=db.backref('yorumlar', lazy=True))

# Çoka-çok ilişki tablosu
gonderi_kategori = db.Table('gonderi_kategori',
    db.Column('gonderi_id', db.Integer, db.ForeignKey('gonderi.id'), primary_key=True),
    db.Column('kategori_id', db.Integer, db.ForeignKey('kategori.id'), primary_key=True)
)
\`\`\`

## Alıştırmalar

1. **Blog Uygulaması Geliştirme**
   - Kullanıcı kaydı ve girişi ekleyin
   - Yorum sistemi oluşturun
   - Kategorileri yönetin
   - Gönderi arama özelliği ekleyin

2. **API Geliştirme**
   - RESTful API endpoint'leri oluşturun
   - JWT authentication ekleyin
   - API dokümantasyonu hazırlayın
   - Rate limiting uygulayın

3. **Performans İyileştirmeleri**
   - Caching mekanizması ekleyin
   - Veritabanı sorgularını optimize edin
   - Statik dosyaları sıkıştırın
   - Yük testi yapın

## Sonraki Adımlar

1. [Django ile Web Geliştirme](/topics/python/web-gelistirme/django)
2. [FastAPI ile Modern API Geliştirme](/topics/python/web-gelistirme/fastapi)
3. [Web Deployment ve DevOps](/topics/python/web-gelistirme/deployment)

## Faydalı Kaynaklar

- [Flask Resmi Dokümantasyonu](https://flask.palletsprojects.com/)
- [Flask-SQLAlchemy Dokümantasyonu](https://flask-sqlalchemy.palletsprojects.com/)
- [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
- [Flask Extensions Registry](https://flask.palletsprojects.com/en/2.0.x/extensions/)
`;

const learningPath = [
  {
    title: '1. Flask Temelleri',
    description: 'Flask\'in temel kavramlarını ve yapısını öğrenin.',
    topics: [
      'Flask kurulumu ve konfigürasyonu',
      'Routing ve URL yapısı',
      'Request ve Response nesneleri',
      'Şablon sistemi (Jinja2)',
      'Statik dosya yönetimi',
    ],
    icon: '🌱',
    href: '/topics/python/web-gelistirme/flask/temeller'
  },
  {
    title: '2. Veritabanı Entegrasyonu',
    description: 'Flask-SQLAlchemy ile veritabanı işlemlerini öğrenin.',
    topics: [
      'Model tanımlama',
      'CRUD işlemleri',
      'İlişkisel veritabanı',
      'Migrations',
      'Query optimizasyonu',
    ],
    icon: '💾',
    href: '/topics/python/web-gelistirme/flask/veritabani'
  },
  {
    title: '3. Kullanıcı Yönetimi',
    description: 'Kullanıcı kimlik doğrulama ve yetkilendirme sistemleri.',
    topics: [
      'Kullanıcı kaydı ve girişi',
      'Session yönetimi',
      'Password hashing',
      'Role-based access control',
      'OAuth entegrasyonu',
    ],
    icon: '🔐',
    href: '/topics/python/web-gelistirme/flask/auth'
  },
  {
    title: '4. API Geliştirme',
    description: 'Flask ile RESTful API\'lar geliştirin.',
    topics: [
      'RESTful endpoint tasarımı',
      'Serialization/Deserialization',
      'API authentication',
      'Rate limiting',
      'API dokümantasyonu',
    ],
    icon: '🔌',
    href: '/topics/python/web-gelistirme/flask/api'
  }
];

export default function FlaskPage() {
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