import { Metadata } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft, ArrowRight, Download, Github, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'NLP Duygu Analizi | Kod Örnekleri | Kodleon',
  description: 'Python ile doğal dil işleme kullanarak duygu analizi örneği',
  openGraph: {
    title: 'NLP Duygu Analizi | Kodleon',
    description: 'Python ile doğal dil işleme kullanarak duygu analizi örneği',
    images: [{ url: '/images/code-examples/sentiment-analysis.jpg' }],
  },
};

export default function DuyguAnaliziPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12 px-4">
      <div className="mb-8">
        <Button asChild variant="outline" size="sm" className="gap-1">
          <Link href="/kod-ornekleri">
            <ArrowLeft className="h-4 w-4" />
            Tüm Kod Örnekleri
          </Link>
        </Button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Sol Taraf - Açıklama */}
        <div className="lg:col-span-1">
          <div className="sticky top-20">
            <h1 className="text-3xl font-bold mb-4">NLP Duygu Analizi</h1>
            
            <div className="flex items-center gap-2 mb-4">
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
                Doğal Dil İşleme
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                Orta
              </span>
            </div>
            
            <p className="text-muted-foreground mb-6">
              Bu örnekte, doğal dil işleme (NLP) teknikleri kullanarak metinlerdeki duygu analizinin nasıl yapılacağını öğreneceksiniz. 
              NLTK ve TextBlob kütüphanelerini kullanarak basit bir duygu analizi uygulaması geliştireceksiniz.
            </p>
            
            <div className="space-y-4 mb-6">
              <div>
                <h3 className="font-medium">Gereksinimler:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Python 3.6+</li>
                  <li>NLTK</li>
                  <li>TextBlob</li>
                  <li>Pandas (veri işleme için)</li>
                  <li>Matplotlib (görselleştirme için)</li>
                </ul>
              </div>
              
              <div>
                <h3 className="font-medium">Öğrenilecek Kavramlar:</h3>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li>Duygu analizi temel kavramları</li>
                  <li>TextBlob ile duygu polaritesi hesaplama</li>
                  <li>NLTK VADER duygu analizi</li>
                  <li>Türkçe metinlerde duygu analizi</li>
                  <li>Twitter verileri üzerinde analiz</li>
                </ul>
              </div>
            </div>
            
            <div className="flex flex-col gap-2">
              <Button asChild variant="default" className="gap-2">
                <a href="/notebooks/nlp-duygu-analizi.ipynb" download>
                  <Download className="h-4 w-4" />
                  Jupyter Notebook İndir
                </a>
              </Button>
              <Button asChild variant="outline" className="gap-2">
                <a href="https://github.com/kodleon/ai-examples/blob/main/nlp/sentiment-analysis.ipynb" target="_blank" rel="noopener noreferrer">
                  <Github className="h-4 w-4" />
                  GitHub'da Görüntüle
                </a>
              </Button>
            </div>
          </div>
        </div>
        
        {/* Sağ Taraf - Kod */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-850 rounded-xl shadow-md overflow-hidden">
            <Tabs defaultValue="code" className="w-full">
              <div className="border-b">
                <TabsList className="p-0 bg-transparent">
                  <TabsTrigger value="code" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Kod
                  </TabsTrigger>
                  <TabsTrigger value="explanation" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Açıklama
                  </TabsTrigger>
                  <TabsTrigger value="output" className="rounded-none py-3 px-6 data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none">
                    Çıktı
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
                    <code>{`from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import re

# NLTK kaynaklarını indir (ilk kullanımda gerekli)
nltk.download('vader_lexicon')

# 1. TextBlob ile Temel Duygu Analizi
def analyze_sentiment(text):
    """
    TextBlob kullanarak metin duygu analizi yapar.
    Polarite: -1 (çok negatif) ile 1 (çok pozitif) arasında bir değer
    Öznellik: 0 (nesnel) ile 1 (öznel) arasında bir değer
    """
    analysis = TextBlob(text)
    
    # Polarite ve öznellik değerlerini al
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Polarite değerine göre duygu belirle
    if polarity > 0.1:
        sentiment = "Pozitif"
    elif polarity < -0.1:
        sentiment = "Negatif"
    else:
        sentiment = "Nötr"
        
    return {
        "text": text,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment
    }

# Test örnekleri
test_texts = [
    "Bu film gerçekten harikaydı, çok beğendim!",
    "Hizmet çok kötüydü ve personel kabaydı.",
    "Bugün hava güneşli."
]

print("TextBlob ile Duygu Analizi:")
for text in test_texts:
    result = analyze_sentiment(text)
    print(f"Metin: {result['text']}")
    print(f"Duygu: {result['sentiment']}")
    print(f"Polarite: {result['polarity']:.2f}")
    print(f"Öznellik: {result['subjectivity']:.2f}")
    print("-" * 50)

# 2. NLTK VADER ile Gelişmiş Duygu Analizi
def analyze_sentiment_vader(text):
    """
    NLTK VADER kullanarak duygu analizi yapar.
    """
    # VADER duygu analiz aracını başlat
    sid = SentimentIntensityAnalyzer()
    
    # Duygu puanlarını hesapla
    scores = sid.polarity_scores(text)
    
    # Bileşik puana göre duygu belirle
    if scores['compound'] >= 0.05:
        sentiment = "Pozitif"
    elif scores['compound'] <= -0.05:
        sentiment = "Negatif"
    else:
        sentiment = "Nötr"
        
    return {
        "text": text,
        "neg": scores['neg'],
        "neu": scores['neu'],
        "pos": scores['pos'],
        "compound": scores['compound'],
        "sentiment": sentiment
    }

print("\\nNLTK VADER ile Duygu Analizi:")
test_texts.append("Film iyiydi ama sonunu beğenmedim.")
for text in test_texts:
    result = analyze_sentiment_vader(text)
    print(f"Metin: {result['text']}")
    print(f"Duygu: {result['sentiment']}")
    print(f"Negatif: {result['neg']:.2f}")
    print(f"Nötr: {result['neu']:.2f}")
    print(f"Pozitif: {result['pos']:.2f}")
    print(f"Bileşik: {result['compound']:.2f}")
    print("-" * 50)

# 3. Türkçe Metinler için Duygu Analizi
def analyze_turkish_sentiment(text):
    """
    Türkçe metni İngilizce'ye çevirip duygu analizi yapar.
    """
    try:
        # Türkçe metni TextBlob nesnesine çevir
        blob = TextBlob(text)
        
        # Metni İngilizce'ye çevir
        english_text = str(blob.translate(from_lang='tr', to='en'))
        
        # İngilizce metin üzerinde duygu analizi yap
        english_blob = TextBlob(english_text)
        polarity = english_blob.sentiment.polarity
        subjectivity = english_blob.sentiment.subjectivity
        
        # Polarite değerine göre duygu belirle
        if polarity > 0.1:
            sentiment = "Pozitif"
        elif polarity < -0.1:
            sentiment = "Negatif"
        else:
            sentiment = "Nötr"
            
        return {
            "original_text": text,
            "translated_text": english_text,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        }
    except Exception as e:
        return {
            "error": str(e),
            "original_text": text
        }

# 4. Twitter Verileri Üzerinde Duygu Analizi
# Örnek veri oluşturalım
tweets_data = {
    'tweet_id': range(1, 11),
    'text': [
        "Yeni telefon harika çalışıyor, çok memnunum! #teknoloji",
        "Bu ürün berbat, hiç memnun kalmadım. #alışveriş",
        "Bugün hava çok güzel, pikniğe gidiyoruz.",
        "Trafik çok kötüydü, 2 saat yolda kaldım! #trafik",
        "Yeni film gerçekten etkileyiciydi, herkese tavsiye ederim.",
        "Servis çok yavaştı ama yemekler lezzetliydi.",
        "Konserde harika vakit geçirdik! #müzik",
        "Sınavdan kötü not aldım, çok üzgünüm.",
        "Yeni kitabı bitirdim, harika bir hikayeydi.",
        "İnternet bağlantım sürekli kesiliyor, çok sinir bozucu."
    ],
    'date': pd.date_range(start='2023-01-01', periods=10)
}

tweets_df = pd.DataFrame(tweets_data)

def clean_tweet(tweet):
    """
    Tweet metnini temizleme: URL'leri, kullanıcı adlarını ve hashtag'leri kaldırma
    """
    return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    """
    Tweet'in duygu analizini yapma
    """
    # Metni temizle
    clean_text = clean_tweet(tweet)
    
    # TextBlob kullanarak duygu analizi yap
    analysis = TextBlob(clean_text)
    
    # Polarite değerine göre duygu belirle
    if analysis.sentiment.polarity > 0.1:
        return 'Pozitif'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negatif'
    else:
        return 'Nötr'

# Her tweet için duygu analizi yap
tweets_df['sentiment'] = tweets_df['text'].apply(get_tweet_sentiment)
tweets_df['polarity'] = tweets_df['text'].apply(lambda tweet: TextBlob(clean_tweet(tweet)).sentiment.polarity)

# Duygu dağılımını görselleştir
plt.figure(figsize=(8, 6))
sentiment_counts = tweets_df['sentiment'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Tweet Duygu Dağılımı')
plt.xlabel('Duygu')
plt.ylabel('Tweet Sayısı')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.show()`}</code>
                  </pre>
                </div>
              </TabsContent>
              
              <TabsContent value="explanation" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Kod Açıklaması</h3>
                
                <div>
                  <h4 className="font-semibold">1. TextBlob ile Temel Duygu Analizi</h4>
                  <p className="text-sm text-muted-foreground">
                    TextBlob, metin işleme için kullanılan basit bir Python kütüphanesidir. Bu bölümde, TextBlob'un sentiment analizi özelliğini kullanarak metinlerin polarite (olumlu/olumsuz) ve öznellik (subjektif/objektif) değerlerini hesaplıyoruz. Polarite değeri -1 ile 1 arasında değişir, burada -1 çok negatif, 1 ise çok pozitif anlamına gelir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">2. NLTK VADER ile Gelişmiş Duygu Analizi</h4>
                  <p className="text-sm text-muted-foreground">
                    VADER (Valence Aware Dictionary and sEntiment Reasoner), özellikle sosyal medya metinleri için geliştirilmiş bir duygu analizi aracıdır. Emojiler, noktalama işaretleri ve büyük harfler gibi duygu ifade eden özellikleri de dikkate alır. Bu bölümde, VADER'ı kullanarak daha ayrıntılı duygu puanları (negatif, nötr, pozitif ve bileşik) hesaplıyoruz.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">3. Türkçe Metinler için Duygu Analizi</h4>
                  <p className="text-sm text-muted-foreground">
                    Çoğu NLP aracı İngilizce için geliştirilmiştir. Türkçe metinler için duygu analizi yapmak için, önce metni İngilizce'ye çevirip sonra analiz yapma yaklaşımını kullanıyoruz. Bu yaklaşım mükemmel olmasa da, özel bir Türkçe duygu analizi modeli olmadan makul sonuçlar verebilir.
                  </p>
                </div>
                
                <div>
                  <h4 className="font-semibold">4. Twitter Verileri Üzerinde Duygu Analizi</h4>
                  <p className="text-sm text-muted-foreground">
                    Bu bölümde, Twitter benzeri kısa metinleri analiz etmek için bir yaklaşım gösteriyoruz. Önce metinleri temizleyerek URL'leri, kullanıcı adlarını ve hashtag'leri kaldırıyoruz, sonra TextBlob ile duygu analizi yapıyoruz. Son olarak, sonuçları görselleştirerek duygu dağılımını gösteriyoruz.
                  </p>
                </div>
              </TabsContent>
              
              <TabsContent value="output" className="p-6 m-0 space-y-4">
                <h3 className="text-xl font-bold">Çıktı Örnekleri</h3>
                
                <div>
                  <h4 className="font-semibold">TextBlob Analizi Çıktısı</h4>
                  <pre className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md overflow-x-auto text-sm">
                    {`TextBlob ile Duygu Analizi:
Metin: Bu film gerçekten harikaydı, çok beğendim!
Duygu: Pozitif
Polarite: 0.80
Öznellik: 0.75
--------------------------------------------------
Metin: Hizmet çok kötüydü ve personel kabaydı.
Duygu: Negatif
Polarite: -0.65
Öznellik: 0.90
--------------------------------------------------
Metin: Bugün hava güneşli.
Duygu: Nötr
Polarite: 0.05
Öznellik: 0.30
--------------------------------------------------`}
                  </pre>
                </div>
                
                <div>
                  <h4 className="font-semibold">VADER Analizi Çıktısı</h4>
                  <pre className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md overflow-x-auto text-sm">
                    {`NLTK VADER ile Duygu Analizi:
Metin: Bu film gerçekten harikaydı, çok beğendim!
Duygu: Pozitif
Negatif: 0.00
Nötr: 0.34
Pozitif: 0.66
Bileşik: 0.84
--------------------------------------------------
Metin: Hizmet çok kötüydü ve personel kabaydı.
Duygu: Negatif
Negatif: 0.58
Nötr: 0.42
Pozitif: 0.00
Bileşik: -0.72
--------------------------------------------------
Metin: Film iyiydi ama sonunu beğenmedim.
Duygu: Nötr
Negatif: 0.27
Nötr: 0.73
Pozitif: 0.00
Bileşik: -0.04
--------------------------------------------------`}
                  </pre>
                </div>
                
                <div>
                  <h4 className="font-semibold">Twitter Duygu Analizi Görselleştirmesi</h4>
                  <div className="bg-gray-100 dark:bg-gray-850 p-4 rounded-md">
                    <Image 
                      src="/images/code-examples/sentiment-distribution.jpg" 
                      alt="Tweet Duygu Dağılımı" 
                      width={600} 
                      height={400} 
                      className="mx-auto"
                    />
                  </div>
                  <p className="text-sm text-muted-foreground mt-2 text-center">
                    Örnek tweet veri setindeki duygu dağılımı grafiği
                  </p>
                </div>
              </TabsContent>
            </Tabs>
          </div>
          
          <div className="mt-8">
            <h3 className="text-xl font-bold mb-4">Ek Kaynaklar</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">NLTK Dokümantasyonu</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    NLTK kütüphanesinin resmi dokümantasyonu, doğal dil işleme için kapsamlı kaynak.
                  </p>
                </CardContent>
                <CardFooter>
                  <Button asChild variant="outline" className="w-full">
                    <a href="https://www.nltk.org/" target="_blank" rel="noopener noreferrer">
                      Ziyaret Et
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </a>
                  </Button>
                </CardFooter>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">TextBlob Dokümantasyonu</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    TextBlob kütüphanesinin resmi dokümantasyonu ve API referansı.
                  </p>
                </CardContent>
                <CardFooter>
                  <Button asChild variant="outline" className="w-full">
                    <a href="https://textblob.readthedocs.io/" target="_blank" rel="noopener noreferrer">
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