import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: "Python'a Giriş | Python Temelleri | Kodleon",
  description: "Python programlama dilinin temel kavramlarını, kurulum adımlarını ve geliştirme ortamı hazırlığını öğrenin.",
};

const content = `
# Python'a Giriş

Python, basit ve anlaşılır sözdizimi ile öğrenmesi kolay, güçlü ve çok yönlü bir programlama dilidir. Bu bölümde Python'un temel kavramlarını ve kurulum adımlarını öğreneceksiniz.

## Python Nedir?

Python, Guido van Rossum tarafından 1991 yılında geliştirilmiş, yüksek seviyeli, yorumlamalı bir programlama dilidir. Öne çıkan özellikleri:

- **Okunabilir Sözdizimi**: Girintileme tabanlı, açık ve anlaşılır kod yapısı
- **Çok Yönlülük**: Web geliştirme, veri bilimi, yapay zeka, otomasyon ve daha fazlası
- **Geniş Kütüphane**: Zengin standart kütüphane ve üçüncü parti paketler
- **Platform Bağımsız**: Windows, macOS, Linux ve diğer platformlarda çalışır
- **Ücretsiz ve Açık Kaynak**: Özgürce kullanılabilir ve geliştirilebilir

## Kurulum Adımları

### Windows için Kurulum

1. [Python İndirme Sayfası](https://www.python.org/downloads/)'nı ziyaret edin
2. En son Python sürümünü indirin
3. Kurulum dosyasını çalıştırın
4. "Add Python to PATH" seçeneğini işaretleyin
5. "Install Now" ile kurulumu başlatın

\`\`\`bash
# Kurulumu doğrulamak için:
python --version
pip --version
\`\`\`

### macOS için Kurulum

macOS genellikle Python ile birlikte gelir, ancak güncel sürüm için:

1. Homebrew kullanarak:
\`\`\`bash
brew install python
\`\`\`

2. Ya da Python web sitesinden .pkg dosyasını indirin

### Linux için Kurulum

Çoğu Linux dağıtımı Python ile birlikte gelir. Güncel sürüm için:

\`\`\`bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# Fedora
sudo dnf install python3 python3-pip
\`\`\`

## Geliştirme Ortamı Hazırlığı

### 1. Metin Editörü / IDE Seçimi

Popüler seçenekler:

- **Visual Studio Code**: Hafif, özelleştirilebilir, ücretsiz
- **PyCharm**: Güçlü, profesyonel, Python odaklı
- **Jupyter Notebook**: Veri bilimi ve interaktif geliştirme
- **Sublime Text**: Hızlı ve hafif metin editörü

### 2. VS Code Python Eklentileri

Önerilen eklentiler:

- Python (Microsoft)
- Pylance
- Python Indent
- Python Docstring Generator
- Python Test Explorer

### 3. Sanal Ortam Oluşturma

Proje bağımlılıklarını izole etmek için sanal ortam kullanımı önerilir:

\`\`\`bash
# Sanal ortam oluşturma
python -m venv myenv

# Windows'ta aktifleştirme
myenv\\Scripts\\activate

# macOS/Linux'ta aktifleştirme
source myenv/bin/activate

# Paket yükleme
pip install package_name
\`\`\`

## İlk Python Programı

1. Yeni bir dosya oluşturun: \`hello.py\`

\`\`\`python
# İlk Python programımız
print("Merhaba, Python!")

# Basit bir hesaplama
x = 5
y = 3
print(f"{x} + {y} = {x + y}")

# Kullanıcıdan girdi alma
isim = input("Adınız nedir? ")
print(f"Merhaba, {isim}!")
\`\`\`

2. Programı çalıştırma:

\`\`\`bash
python hello.py
\`\`\`

## Python Interpreter

Python interpreter'ı interaktif bir kabuk olarak kullanabilirsiniz:

\`\`\`python
# Terminal/Komut İstemcisinde:
python

>>> print("Merhaba!")
Merhaba!
>>> 2 + 3
5
>>> exit()  # Çıkış için
\`\`\`

## Kod Yazma Kuralları

1. **Girintileme**: 4 boşluk veya 1 tab (4 boşluk önerilir)
2. **Satır Uzunluğu**: 79 karakter (PEP 8)
3. **İsimlendirme**:
   - Değişkenler: snake_case (örn. \`user_name\`)
   - Sınıflar: PascalCase (örn. \`UserAccount\`)
   - Sabitler: UPPER_CASE (örn. \`MAX_VALUE\`)

## Alıştırmalar

1. **Kurulum Kontrolü**
   - Python ve pip sürümlerini kontrol edin
   - Sanal ortam oluşturun ve aktifleştirin
   - Bir paket yükleyin ve import edin

2. **İlk Program**
   - Kullanıcıdan iki sayı alan
   - Bu sayıların toplamını, farkını, çarpımını ve bölümünü hesaplayan
   - Sonuçları düzenli bir şekilde ekrana yazdıran bir program yazın

3. **IDE Alıştırması**
   - VS Code'da Python eklentilerini kurun
   - Kod tamamlama ve hata ayıklama özelliklerini deneyin
   - Bir Python dosyası oluşturup çalıştırın

## Sonraki Adımlar

- [Değişkenler ve Veri Tipleri](/topics/python/temel-python/degiskenler-ve-veri-tipleri)
- [Python Resmi Dokümantasyonu](https://docs.python.org/)
- [VS Code Python Eğitimi](https://code.visualstudio.com/docs/python/python-tutorial)
`;

export default function PythonIntroductionPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/temel-python" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Python Temelleri
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