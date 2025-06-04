import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'NumPy ile Görüntü İşleme | Python Veri Bilimi | Kodleon',
  description: 'NumPy kullanarak görüntü işleme, filtreleme, dönüşüm ve analiz tekniklerini öğrenin.',
};

const content = `
# NumPy ile Görüntü İşleme

NumPy, görüntüleri sayısal diziler olarak temsil ederek güçlü görüntü işleme yetenekleri sunar. Bu bölümde temel görüntü işleme tekniklerini ve uygulamalarını inceleyeceğiz.

## Görüntüleri Yükleme ve Temel İşlemler

Görüntüleri NumPy dizilerine dönüştürme ve temel işlemler:

\`\`\`python
import numpy as np
from PIL import Image  # Görüntü yükleme için
import matplotlib.pyplot as plt  # Görüntüleme için

# Görüntüyü yükle
img = Image.open('ornek.jpg')
img_array = np.array(img)

# Görüntü bilgileri
print("Boyut:", img_array.shape)
print("Veri tipi:", img_array.dtype)
print("Minimum piksel değeri:", img_array.min())
print("Maksimum piksel değeri:", img_array.max())

# Gri tonlamalı görüntüye dönüştürme
if len(img_array.shape) == 3:  # Renkli görüntü ise
    gray = np.mean(img_array, axis=2).astype(np.uint8)
else:
    gray = img_array

# Görüntüyü göster
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img_array), plt.title('Orijinal')
plt.subplot(122), plt.imshow(gray, cmap='gray'), plt.title('Gri Tonlamalı')
plt.show()
\`\`\`

## Görüntü Manipülasyonu

Temel görüntü manipülasyon teknikleri:

\`\`\`python
# Parlaklık ayarlama
brightness = 50
brightened = np.clip(img_array + brightness, 0, 255).astype(np.uint8)

# Kontrast ayarlama
contrast = 1.5
contrasted = np.clip(img_array * contrast, 0, 255).astype(np.uint8)

# Görüntüyü döndürme
rotated = np.rot90(img_array)  # 90 derece döndürme

# Görüntüyü yeniden boyutlandırma
from skimage.transform import resize
new_size = (img_array.shape[0]//2, img_array.shape[1]//2)
resized = resize(img_array, new_size, anti_aliasing=True)
resized = (resized * 255).astype(np.uint8)

# Görüntüyü aynalama
flipped_h = np.fliplr(img_array)  # Yatay aynalama
flipped_v = np.flipud(img_array)  # Dikey aynalama
\`\`\`

## Görüntü Filtreleme

Temel filtreleme işlemleri:

\`\`\`python
# Gaussian bulanıklaştırma
def gaussian_kernel(size, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, size),
                       np.linspace(-1, 1, size))
    d = np.sqrt(x*x + y*y)
    g = np.exp(-(d**2)/(2.0*sigma**2))
    return g / g.sum()

# Konvolüsyon işlemi
def convolve2d(image, kernel):
    output = np.zeros_like(image)
    pad_width = kernel.shape[0] // 2
    padded = np.pad(image, pad_width, mode='edge')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(
                padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel
            )
    return output

# Filtreleri uygula
kernel_size = 5
gaussian_blur = gaussian_kernel(kernel_size, 1.0)
blurred = convolve2d(gray, gaussian_blur)

# Kenar tespiti için Sobel filtreleri
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

edges_x = convolve2d(gray, sobel_x)
edges_y = convolve2d(gray, sobel_y)
edges = np.sqrt(edges_x**2 + edges_y**2)
\`\`\`

## Histogram İşlemleri

Görüntü histogramı analizi ve işlemleri:

\`\`\`python
# Histogram hesaplama
hist, bins = np.histogram(gray, bins=256, range=(0, 256))

# Histogram eşitleme
def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[img]

# Histogram eşitlemeyi uygula
equalized = histogram_equalization(gray)

# Sonuçları görselleştir
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(gray, cmap='gray'), plt.title('Orijinal')
plt.subplot(132), plt.hist(gray.flatten(), 256, [0, 256]), plt.title('Histogram')
plt.subplot(133), plt.imshow(equalized, cmap='gray'), plt.title('Eşitlenmiş')
plt.show()
\`\`\`

## Morfolojik İşlemler

Temel morfolojik operasyonlar:

\`\`\`python
# İkili görüntü oluşturma (thresholding)
threshold = 127
binary = (gray > threshold).astype(np.uint8) * 255

# Yapısal element oluşturma
kernel = np.ones((3, 3), np.uint8)

# Genişletme (Dilation)
def dilate(image, kernel):
    output = np.zeros_like(image)
    pad_width = kernel.shape[0] // 2
    padded = np.pad(image, pad_width, mode='constant')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.max(window * kernel)
    return output

# Aşındırma (Erosion)
def erode(image, kernel):
    output = np.zeros_like(image)
    pad_width = kernel.shape[0] // 2
    padded = np.pad(image, pad_width, mode='constant', constant_values=255)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.min(window * kernel)
    return output

# Morfolojik işlemleri uygula
dilated = dilate(binary, kernel)
eroded = erode(binary, kernel)
\`\`\`

## Alıştırmalar

1. **Temel Görüntü İşleme**
   - Bir görüntüyü yükleyin ve gri tonlamalıya dönüştürün
   - Parlaklık ve kontrast ayarlamalarını uygulayın
   - Sonuçları karşılaştırın

2. **Filtreleme ve Kenar Tespiti**
   - Farklı boyutlarda Gaussian filtreleri oluşturun
   - Sobel filtrelerini uygulayın
   - Kenar tespiti sonuçlarını iyileştirin

3. **Histogram İşlemleri**
   - Bir görüntünün histogramını hesaplayın
   - Histogram eşitleme uygulayın
   - Sonuçları görselleştirin

## Sonraki Adımlar

1. [Pandas ile Veri Analizi](/topics/python/veri-bilimi/pandas)
2. [Matplotlib ile Veri Görselleştirme](/topics/python/veri-bilimi/matplotlib)
3. [Scikit-learn ile Makine Öğrenmesi](/topics/python/veri-bilimi/scikit-learn)

## Faydalı Kaynaklar

- [Scikit-image Dokümantasyonu](https://scikit-image.org/)
- [OpenCV-Python Eğitimleri](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Digital Image Processing](https://www.imageprocessingplace.com/)
`;

export default function NumPyImageProcessingPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/veri-bilimi/numpy" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              NumPy
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