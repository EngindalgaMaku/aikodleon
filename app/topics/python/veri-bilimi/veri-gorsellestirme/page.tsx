import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Veri Görselleştirme | Python Veri Bilimi | Kodleon',
  description: 'Python ile veri görselleştirme teknikleri. Matplotlib, Seaborn ve Plotly kullanarak etkileşimli grafikler oluşturma.',
};

const content = `
# Veri Görselleştirme

Bu bölümde, Python'da veri görselleştirme tekniklerini ve popüler kütüphaneleri öğreneceğiz.

## Matplotlib ile Temel Grafikler

\`\`\`python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
import seaborn as sns
from dataclasses import dataclass

@dataclass
class GrafikAyarlari:
    baslik: str
    x_etiketi: str
    y_etiketi: str
    boyut: Tuple[int, int] = (10, 6)
    stil: str = 'seaborn'
    renk_paleti: str = 'viridis'
    font_boyutu: int = 12
    grid: bool = True

class TemelGrafikler:
    def __init__(self, ayarlar: GrafikAyarlari):
        self.ayarlar = ayarlar
        plt.style.use(ayarlar.stil)
        plt.rcParams['font.size'] = ayarlar.font_boyutu
        
    def _grafik_hazirla(self):
        """Grafik için temel ayarları yapar"""
        plt.figure(figsize=self.ayarlar.boyut)
        if self.ayarlar.grid:
            plt.grid(True, linestyle='--', alpha=0.7)
            
    def _grafik_bitir(self):
        """Grafik başlık ve etiketlerini ekler"""
        plt.title(self.ayarlar.baslik)
        plt.xlabel(self.ayarlar.x_etiketi)
        plt.ylabel(self.ayarlar.y_etiketi)
        plt.tight_layout()
        
    def cizgi_grafigi(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      etiket: Optional[str] = None,
                      stil: str = '-') -> None:
        """Çizgi grafiği çizer"""
        self._grafik_hazirla()
        plt.plot(x, y, stil, label=etiket)
        if etiket:
            plt.legend()
        self._grafik_bitir()
        
    def sacilim_grafigi(self,
                       x: np.ndarray,
                       y: np.ndarray,
                       boyut: Optional[np.ndarray] = None,
                       renk: Optional[np.ndarray] = None,
                       alfa: float = 0.6) -> None:
        """Saçılım grafiği çizer"""
        self._grafik_hazirla()
        plt.scatter(x, y, s=boyut, c=renk, alpha=alfa)
        if renk is not None:
            plt.colorbar(label='Değer')
        self._grafik_bitir()
        
    def histogram(self,
                 veri: np.ndarray,
                 aralik_sayisi: int = 30,
                 yogunluk: bool = False,
                 kenar_rengi: str = 'black') -> None:
        """Histogram çizer"""
        self._grafik_hazirla()
        plt.hist(veri, bins=aralik_sayisi, density=yogunluk,
                edgecolor=kenar_rengi, alpha=0.7)
        self._grafik_bitir()
        
    def kutu_grafigi(self,
                    veri: List[np.ndarray],
                    etiketler: Optional[List[str]] = None) -> None:
        """Kutu grafiği çizer"""
        self._grafik_hazirla()
        plt.boxplot(veri, labels=etiketler)
        self._grafik_bitir()
        
    def pasta_grafigi(self,
                     degerler: np.ndarray,
                     etiketler: List[str],
                     patlama: Optional[List[float]] = None) -> None:
        """Pasta grafiği çizer"""
        self._grafik_hazirla()
        plt.pie(degerler, labels=etiketler, explode=patlama,
               autopct='%1.1f%%', shadow=True)
        self._grafik_bitir()

# Kullanım örneği
if __name__ == "__main__":
    # Örnek veri oluştur
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Grafik ayarlarını tanımla
    ayarlar = GrafikAyarlari(
        baslik="Sinüs ve Kosinüs Fonksiyonları",
        x_etiketi="x",
        y_etiketi="y",
        boyut=(12, 6)
    )
    
    # Grafik çizici oluştur
    grafik = TemelGrafikler(ayarlar)
    
    # Çizgi grafiği
    grafik.cizgi_grafigi(x, y1, etiket="sin(x)")
    grafik.cizgi_grafigi(x, y2, etiket="cos(x)")
    plt.show()
    
    # Saçılım grafiği
    x_scatter = np.random.randn(100)
    y_scatter = np.random.randn(100)
    boyutlar = np.random.rand(100) * 200
    renkler = np.random.rand(100)
    
    ayarlar.baslik = "Renkli Saçılım Grafiği"
    grafik = TemelGrafikler(ayarlar)
    grafik.sacilim_grafigi(x_scatter, y_scatter, boyutlar, renkler)
    plt.show()
\`\`\`

## Seaborn ile İstatistiksel Grafikler

\`\`\`python
import seaborn as sns
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class SeabornAyarlari:
    baslik: str
    x_degiskeni: str
    y_degiskeni: str
    hue_degiskeni: Optional[str] = None
    stil: str = 'whitegrid'
    palet: str = 'husl'
    boyut: Tuple[int, int] = (10, 6)

class IstatistikselGrafikler:
    def __init__(self, ayarlar: SeabornAyarlari):
        self.ayarlar = ayarlar
        sns.set_style(ayarlar.stil)
        sns.set_palette(ayarlar.palet)
        
    def _grafik_hazirla(self):
        """Grafik için temel ayarları yapar"""
        plt.figure(figsize=self.ayarlar.boyut)
        
    def dagilim_grafigi(self,
                       veri: pd.DataFrame,
                       kestirim: bool = True) -> None:
        """Dağılım grafiği çizer"""
        self._grafik_hazirla()
        sns.jointplot(
            data=veri,
            x=self.ayarlar.x_degiskeni,
            y=self.ayarlar.y_degiskeni,
            hue=self.ayarlar.hue_degiskeni,
            kind='scatter',
            height=8
        )
        plt.suptitle(self.ayarlar.baslik, y=1.02)
        
    def violin_grafigi(self,
                      veri: pd.DataFrame,
                      ic_grafik: str = 'box') -> None:
        """Violin grafiği çizer"""
        self._grafik_hazirla()
        sns.violinplot(
            data=veri,
            x=self.ayarlar.x_degiskeni,
            y=self.ayarlar.y_degiskeni,
            hue=self.ayarlar.hue_degiskeni,
            inner=ic_grafik
        )
        plt.title(self.ayarlar.baslik)
        
    def isı_haritasi(self,
                    veri: pd.DataFrame,
                    korelasyon: bool = True,
                    format: str = '.2f') -> None:
        """Isı haritası çizer"""
        self._grafik_hazirla()
        if korelasyon:
            veri = veri.corr()
        sns.heatmap(veri, annot=True, fmt=format, cmap='coolwarm')
        plt.title(self.ayarlar.baslik)
        
    def cift_grafigi(self,
                    veri: pd.DataFrame,
                    degiskenler: Optional[List[str]] = None) -> None:
        """Değişken çiftleri için grafik matrisi çizer"""
        self._grafik_hazirla()
        sns.pairplot(
            veri,
            vars=degiskenler,
            hue=self.ayarlar.hue_degiskeni,
            diag_kind='kde'
        )
        plt.suptitle(self.ayarlar.baslik, y=1.02)
        
    def kategori_grafigi(self,
                        veri: pd.DataFrame,
                        ci: Optional[float] = 95) -> None:
        """Kategorik veri grafiği çizer"""
        self._grafik_hazirla()
        sns.catplot(
            data=veri,
            x=self.ayarlar.x_degiskeni,
            y=self.ayarlar.y_degiskeni,
            hue=self.ayarlar.hue_degiskeni,
            kind='bar',
            ci=ci,
            height=6,
            aspect=1.5
        )
        plt.title(self.ayarlar.baslik)

# Kullanım örneği
if __name__ == "__main__":
    # Örnek veri oluştur
    np.random.seed(42)
    
    # Iris veri setini yükle
    iris = sns.load_dataset('iris')
    
    # Grafik ayarlarını tanımla
    ayarlar = SeabornAyarlari(
        baslik="Iris Veri Seti Analizi",
        x_degiskeni="sepal_length",
        y_degiskeni="sepal_width",
        hue_degiskeni="species"
    )
    
    # Grafik çizici oluştur
    grafik = IstatistikselGrafikler(ayarlar)
    
    # Dağılım grafiği
    grafik.dagilim_grafigi(iris)
    plt.show()
    
    # Violin grafiği
    grafik.violin_grafigi(iris)
    plt.show()
    
    # Isı haritası
    grafik.isı_haritasi(iris.drop('species', axis=1))
    plt.show()
    
    # Çift grafiği
    grafik.cift_grafigi(iris)
    plt.show()
\`\`\`

## Plotly ile Etkileşimli Grafikler

\`\`\`python
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class PlotlyAyarlari:
    baslik: str
    x_etiketi: str
    y_etiketi: str
    tema: str = 'plotly'
    renk_paleti: Optional[str] = None
    boyut: Tuple[int, int] = (800, 500)

class EtkileşimliGrafikler:
    def __init__(self, ayarlar: PlotlyAyarlari):
        self.ayarlar = ayarlar
        
    def _grafik_ayarla(self, fig: go.Figure) -> go.Figure:
        """Grafik ayarlarını uygular"""
        fig.update_layout(
            title=self.ayarlar.baslik,
            xaxis_title=self.ayarlar.x_etiketi,
            yaxis_title=self.ayarlar.y_etiketi,
            width=self.ayarlar.boyut[0],
            height=self.ayarlar.boyut[1],
            template=self.ayarlar.tema
        )
        return fig
        
    def cizgi_grafigi(self,
                      veri: pd.DataFrame,
                      renk_degiskeni: Optional[str] = None) -> go.Figure:
        """Etkileşimli çizgi grafiği oluşturur"""
        fig = px.line(
            veri,
            x=self.ayarlar.x_etiketi,
            y=self.ayarlar.y_etiketi,
            color=renk_degiskeni,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        return self._grafik_ayarla(fig)
        
    def sacilim_grafigi(self,
                       veri: pd.DataFrame,
                       boyut_degiskeni: Optional[str] = None,
                       renk_degiskeni: Optional[str] = None,
                       metin_degiskeni: Optional[str] = None) -> go.Figure:
        """Etkileşimli saçılım grafiği oluşturur"""
        fig = px.scatter(
            veri,
            x=self.ayarlar.x_etiketi,
            y=self.ayarlar.y_etiketi,
            size=boyut_degiskeni,
            color=renk_degiskeni,
            hover_name=metin_degiskeni,
            color_continuous_scale=self.ayarlar.renk_paleti
        )
        return self._grafik_ayarla(fig)
        
    def bar_grafigi(self,
                   veri: pd.DataFrame,
                   renk_degiskeni: Optional[str] = None,
                   yığın: bool = False) -> go.Figure:
        """Etkileşimli bar grafiği oluşturur"""
        fig = px.bar(
            veri,
            x=self.ayarlar.x_etiketi,
            y=self.ayarlar.y_etiketi,
            color=renk_degiskeni,
            barmode='stack' if yığın else 'group'
        )
        return self._grafik_ayarla(fig)
        
    def kutu_grafigi(self,
                    veri: pd.DataFrame,
                    renk_degiskeni: Optional[str] = None,
                    nokta_goster: bool = True) -> go.Figure:
        """Etkileşimli kutu grafiği oluşturur"""
        fig = px.box(
            veri,
            x=self.ayarlar.x_etiketi,
            y=self.ayarlar.y_etiketi,
            color=renk_degiskeni,
            points='all' if nokta_goster else 'outliers'
        )
        return self._grafik_ayarla(fig)
        
    def histogram(self,
                 veri: pd.DataFrame,
                 renk_degiskeni: Optional[str] = None,
                 aralik_sayisi: int = 30) -> go.Figure:
        """Etkileşimli histogram oluşturur"""
        fig = px.histogram(
            veri,
            x=self.ayarlar.x_etiketi,
            color=renk_degiskeni,
            nbins=aralik_sayisi,
            marginal='box'
        )
        return self._grafik_ayarla(fig)
        
    def pasta_grafigi(self,
                     veri: pd.DataFrame,
                     degerler: str,
                     isimler: str) -> go.Figure:
        """Etkileşimli pasta grafiği oluşturur"""
        fig = px.pie(
            veri,
            values=degerler,
            names=isimler,
            hole=0.3
        )
        return self._grafik_ayarla(fig)

# Kullanım örneği
if __name__ == "__main__":
    # Örnek veri oluştur
    np.random.seed(42)
    
    # Gapminder veri setini yükle
    gapminder = px.data.gapminder()
    
    # Grafik ayarlarını tanımla
    ayarlar = PlotlyAyarlari(
        baslik="Ülkelerin Yaşam Beklentisi Analizi",
        x_etiketi="Yıl",
        y_etiketi="Yaşam Beklentisi",
        tema="plotly_dark"
    )
    
    # Grafik çizici oluştur
    grafik = EtkileşimliGrafikler(ayarlar)
    
    # Çizgi grafiği
    fig = grafik.cizgi_grafigi(
        gapminder,
        renk_degiskeni="continent"
    )
    fig.show()
    
    # Saçılım grafiği
    ayarlar.x_etiketi = "gdpPercap"
    ayarlar.y_etiketi = "lifeExp"
    ayarlar.baslik = "GSYH ve Yaşam Beklentisi İlişkisi"
    
    fig = grafik.sacilim_grafigi(
        gapminder,
        boyut_degiskeni="pop",
        renk_degiskeni="continent",
        metin_degiskeni="country"
    )
    fig.show()
\`\`\`

## Alıştırmalar

1. **Temel Grafikler**
   - Farklı veri setleri için çizgi ve saçılım grafikleri oluşturun
   - Özel renk paletleri ve stilleri deneyin
   - Alt grafikleri (subplots) kullanarak karşılaştırmalı analizler yapın

2. **İstatistiksel Görselleştirme**
   - Violin ve kutu grafikleri ile dağılım analizleri yapın
   - Isı haritaları ile korelasyon analizleri gerçekleştirin
   - Kategorik veri görselleştirmeleri oluşturun

3. **Etkileşimli Grafikler**
   - Plotly ile animasyonlu grafikler oluşturun
   - Özel araç ipuçları (tooltips) ekleyin
   - Grafikleri HTML olarak kaydedin ve paylaşın

## Sonraki Adımlar

1. [MLOps ve DevOps](/topics/python/veri-bilimi/mlops)
2. [Derin Öğrenme Deployment](/topics/python/veri-bilimi/derin-ogrenme-deployment)
3. [Yapay Zeka Projeleri](/topics/python/veri-bilimi/yapay-zeka-projeleri)

## Faydalı Kaynaklar

- [Matplotlib Dokümantasyonu](https://matplotlib.org/)
- [Seaborn Dokümantasyonu](https://seaborn.pydata.org/)
- [Plotly Dokümantasyonu](https://plotly.com/python/)
- [Python Graph Gallery](https://python-graph-gallery.com/)
`;

const learningPath = [
  {
    title: '1. Matplotlib Temelleri',
    description: 'Python\'da temel veri görselleştirme kütüphanesi olan Matplotlib\'i öğrenin.',
    topics: [
      'Temel grafik oluşturma',
      'Çizgi, çubuk ve pasta grafikleri',
      'Saçılım grafikleri',
      'Grafik özelleştirme',
      'Alt grafikler (subplots)',
    ],
    icon: '📊',
    href: '/topics/python/veri-bilimi/veri-gorsellestirme/matplotlib'
  },
  {
    title: '2. Seaborn ile İstatistiksel Görselleştirme',
    description: 'İstatistiksel veri görselleştirme için Seaborn kütüphanesini keşfedin.',
    topics: [
      'Dağılım grafikleri',
      'Kutu ve violin grafikleri',
      'İlişki grafikleri',
      'Kategorik veri görselleştirme',
      'Isı haritaları',
    ],
    icon: '📈',
    href: '/topics/python/veri-bilimi/veri-gorsellestirme/seaborn'
  },
  {
    title: '3. Plotly ile İnteraktif Grafikler',
    description: 'Web tabanlı interaktif grafikler oluşturmak için Plotly kütüphanesini öğrenin.',
    topics: [
      'İnteraktif çizgi grafikleri',
      'İnteraktif saçılım grafikleri',
      '3D görselleştirme',
      'Animasyonlu grafikler',
      'Dashboard oluşturma',
    ],
    icon: '🔄',
    href: '/topics/python/veri-bilimi/veri-gorsellestirme/plotly'
  },
  {
    title: '4. İleri Düzey Görselleştirme',
    description: 'Karmaşık veri görselleştirme teknikleri ve özel grafik türlerini keşfedin.',
    topics: [
      'Coğrafi veri görselleştirme',
      'Ağ grafikleri',
      'Zaman serisi görselleştirme',
      'Çoklu veri görselleştirme',
      'Özel tema ve stiller',
    ],
    icon: '🎨',
    href: '/topics/python/veri-bilimi/veri-gorsellestirme/ileri-duzey'
  }
];

export default function VeriGorsellestirmePage() {
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