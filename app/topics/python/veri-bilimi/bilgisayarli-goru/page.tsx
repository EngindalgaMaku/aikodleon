import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = {
  title: 'Python ile Bilgisayarlı Görü | Python Veri Bilimi | Kodleon',
  description: 'Python kullanarak görüntü işleme, nesne tespiti ve bilgisayarlı görü uygulamalarını öğrenin.',
};

const content = `
# Python ile Bilgisayarlı Görü

Bilgisayarlı görü, bilgisayarların görüntüleri anlama ve işleme yeteneğini geliştiren bir yapay zeka alt dalıdır. Bu bölümde, Python ile bilgisayarlı görü uygulamalarını öğreneceğiz.

## Temel Görüntü İşleme

### OpenCV ile Görüntü İşleme

\`\`\`python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntü okuma
img = cv2.imread('ornek.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Görüntü boyutlandırma
yeni_boyut = (300, 300)
img_yeniden = cv2.resize(img_rgb, yeni_boyut)

# Gri tonlama
img_gri = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Görüntü bulanıklaştırma
img_blur = cv2.GaussianBlur(img_gri, (5, 5), 0)

# Kenar tespiti
edges = cv2.Canny(img_blur, 100, 200)

# Sonuçları görselleştirme
plt.figure(figsize=(15, 5))
plt.subplot(141), plt.imshow(img_rgb), plt.title('Orijinal')
plt.subplot(142), plt.imshow(img_gri, cmap='gray'), plt.title('Gri')
plt.subplot(143), plt.imshow(img_blur, cmap='gray'), plt.title('Bulanık')
plt.subplot(144), plt.imshow(edges, cmap='gray'), plt.title('Kenarlar')
plt.show()
\`\`\`

### Pillow ile Görüntü İşleme

\`\`\`python
from PIL import Image, ImageEnhance, ImageFilter

# Görüntü açma
img = Image.open('ornek.jpg')

# Renk ayarları
contrast = ImageEnhance.Contrast(img)
img_contrast = contrast.enhance(1.5)

brightness = ImageEnhance.Brightness(img)
img_bright = brightness.enhance(1.2)

# Filtreler uygulama
img_blur = img.filter(ImageFilter.BLUR)
img_sharpen = img.filter(ImageFilter.SHARPEN)

# Döndürme ve yansıtma
img_rotate = img.rotate(45)
img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)

# Görüntüleri kaydetme
img_contrast.save('contrast.jpg')
img_bright.save('bright.jpg')
\`\`\`

## Nesne Tespiti

### YOLO ile Nesne Tespiti

\`\`\`python
import torch
from ultralytics import YOLO

# Model yükleme
model = YOLO('yolov8n.pt')

# Görüntü üzerinde nesne tespiti
sonuclar = model('ornek.jpg')

# Sonuçları görselleştirme
for sonuc in sonuclar:
    boxes = sonuc.boxes
    for box in boxes:
        # Koordinatlar
        x1, y1, x2, y2 = box.xyxy[0]
        # Sınıf
        sinif = box.cls[0]
        # Güven skoru
        skor = box.conf[0]
        print(f"Sınıf: {sinif}, Skor: {skor:.2f}")

# Görselleştirme
sonuclar[0].show()
\`\`\`

### Yüz Tespiti ve Tanıma

\`\`\`python
import cv2
import face_recognition

# Görüntü yükleme
img = cv2.imread('yuzler.jpg')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Yüz tespiti
yuz_konumlari = face_recognition.face_locations(rgb_img)
yuz_kodlamalari = face_recognition.face_encodings(rgb_img, yuz_konumlari)

# Yüzleri işaretleme
for (top, right, bottom, left) in yuz_konumlari:
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

# Sonucu gösterme
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
\`\`\`

## Görüntü Sınıflandırma

### CNN ile Görüntü Sınıflandırma

\`\`\`python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model oluşturma
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Veri artırma
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model eğitimi
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)
\`\`\`

## Görüntü Segmentasyonu

### Semantic Segmentation

\`\`\`python
import torch
import segmentation_models_pytorch as smp

# Model oluşturma
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

# Görüntü ön işleme
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0).float()

# Tahmin
img = cv2.imread('ornek.jpg')
input_tensor = preprocess(img)
mask = model(input_tensor)
mask = torch.sigmoid(mask)
mask = mask.squeeze().detach().numpy()

# Sonucu görselleştirme
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Orijinal Görüntü')
plt.subplot(122), plt.imshow(mask, cmap='gray')
plt.title('Segmentasyon Maskesi')
plt.show()
\`\`\`

## Özellik Çıkarma

### SIFT ve SURF

\`\`\`python
import cv2
import numpy as np

# Görüntü okuma
img = cv2.imread('ornek.jpg', cv2.IMREAD_GRAYSCALE)

# SIFT dedektörü oluşturma
sift = cv2.SIFT_create()

# Anahtar noktaları ve tanımlayıcıları bulma
keypoints, descriptors = sift.detectAndCompute(img, None)

# Anahtar noktaları çizme
img_keypoints = cv2.drawKeypoints(
    img, keypoints, None, 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.imshow(img_keypoints)
plt.title('SIFT Anahtar Noktaları')
plt.show()
\`\`\`

## Optik Karakter Tanıma (OCR)

### Tesseract ile OCR

\`\`\`python
import pytesseract
from PIL import Image

# Görüntü yükleme
img = Image.open('metin.jpg')

# OCR işlemi
text = pytesseract.image_to_string(img, lang='tur')
print("Tanınan Metin:", text)

# Metin bölgelerini bulma
d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

# Metin bölgelerini görselleştirme
img_np = np.array(img)
for i, word in enumerate(d['text']):
    if int(d['conf'][i]) > 60:  # Güven skoru kontrolü
        x, y, w, h = (
            d['left'][i], d['top'][i],
            d['width'][i], d['height'][i]
        )
        cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(img_np)
plt.title('Tespit Edilen Metin Bölgeleri')
plt.show()
\`\`\`

## Alıştırmalar

1. **Temel Görüntü İşleme**
   - Farklı filtreler ve efektler uygulayın
   - Histogram eşitleme deneyin
   - Görüntü birleştirme işlemleri yapın

2. **Nesne Tespiti**
   - Kendi YOLO modelinizi eğitin
   - Özel nesne sınıfları için model geliştirin
   - Gerçek zamanlı nesne tespiti yapın

3. **Görüntü Sınıflandırma**
   - Transfer öğrenme ile model geliştirin
   - Kendi veri setinizi oluşturun
   - Model performansını iyileştirin

## Sonraki Adımlar

1. [Pekiştirmeli Öğrenme](/topics/python/veri-bilimi/pekistirmeli-ogrenme)
2. [Büyük Veri](/topics/python/veri-bilimi/buyuk-veri)
3. [Yapay Zeka Projeleri](/topics/python/veri-bilimi/yapay-zeka-projeleri)

## Faydalı Kaynaklar

- [OpenCV Dokümantasyonu](https://docs.opencv.org/)
- [PyTorch Vision Öğreticileri](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [TensorFlow Görüntü İşleme](https://www.tensorflow.org/tutorials/images)
`;

export default function ComputerVisionPage() {
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

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 