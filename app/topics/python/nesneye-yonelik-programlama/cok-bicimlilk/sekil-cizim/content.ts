export const content = `
# Şekil Çizim Uygulaması

Bu alıştırmada, çok biçimliliği kullanarak farklı geometrik şekilleri çizebilen bir grafik uygulaması geliştireceğiz.

## Problem Tanımı

Farklı geometrik şekilleri çizebilen, taşıyabilen ve boyutlandırabilen bir grafik uygulaması geliştirmemiz gerekiyor. Sistem şu özelliklere sahip olmalı:

* Farklı şekiller için ortak bir arayüz
* Her şekil için özel çizim mantığı
* Renk ve stil özellikleri
* Şekilleri taşıma ve boyutlandırma özellikleri

## Çözüm

### 1. Temel Veri Yapıları

Önce gerekli veri yapılarını tanımlayalım:

\`\`\`python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

# Renk tanımları
class Color(Enum):
    RED = "kırmızı"
    GREEN = "yeşil"
    BLUE = "mavi"
    YELLOW = "sarı"
    BLACK = "siyah"
    WHITE = "beyaz"

# Çizgi stilleri
class LineStyle(Enum):
    SOLID = "düz"
    DASHED = "kesikli"
    DOTTED = "noktalı"

# Nokta veri yapısı
@dataclass
class Point:
    x: float
    y: float
    
    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

# Stil özellikleri
@dataclass
class Style:
    fill_color: Color
    border_color: Color
    line_style: LineStyle
    line_width: float
    
    @staticmethod
    def default() -> 'Style':
        return Style(
            fill_color=Color.WHITE,
            border_color=Color.BLACK,
            line_style=LineStyle.SOLID,
            line_width=1.0
        )
\`\`\`

### 2. Temel Şekil Sınıfı

Tüm şekiller için temel bir soyut sınıf oluşturalım:

\`\`\`python
class Shape(ABC):
    def __init__(self, position: Point, style: Optional[Style] = None):
        self.position = position
        self.style = style or Style.default()
        self.selected = False
    
    @abstractmethod
    def draw(self) -> str:
        pass
    
    @abstractmethod
    def resize(self, scale: float) -> None:
        pass
    
    @abstractmethod
    def get_area(self) -> float:
        pass
    
    @abstractmethod
    def get_perimeter(self) -> float:
        pass
    
    def move(self, dx: float, dy: float) -> None:
        self.position.move(dx, dy)
    
    def select(self) -> None:
        self.selected = True
    
    def deselect(self) -> None:
        self.selected = False
    
    def set_style(self, style: Style) -> None:
        self.style = style
    
    def get_info(self) -> str:
        return f"""
Şekil Bilgisi:
Pozisyon: {self.position}
Alan: {self.get_area():.2f}
Çevre: {self.get_perimeter():.2f}
Dolgu Rengi: {self.style.fill_color.value}
Kenarlık Rengi: {self.style.border_color.value}
Çizgi Stili: {self.style.line_style.value}
Çizgi Kalınlığı: {self.style.line_width}
Seçili: {'Evet' if self.selected else 'Hayır'}
"""
\`\`\`

### 3. Özel Şekil Sınıfları

Farklı geometrik şekiller için özel sınıflar oluşturalım:

\`\`\`python
import math

class Rectangle(Shape):
    def __init__(self, position: Point, width: float, height: float, style: Optional[Style] = None):
        super().__init__(position, style)
        self.width = width
        self.height = height
    
    def draw(self) -> str:
        return f"Dikdörtgen çiziliyor: Pozisyon {self.position}, Genişlik {self.width}, Yükseklik {self.height}"
    
    def resize(self, scale: float) -> None:
        self.width *= scale
        self.height *= scale
    
    def get_area(self) -> float:
        return self.width * self.height
    
    def get_perimeter(self) -> float:
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, position: Point, radius: float, style: Optional[Style] = None):
        super().__init__(position, style)
        self.radius = radius
    
    def draw(self) -> str:
        return f"Daire çiziliyor: Merkez {self.position}, Yarıçap {self.radius}"
    
    def resize(self, scale: float) -> None:
        self.radius *= scale
    
    def get_area(self) -> float:
        return math.pi * self.radius ** 2
    
    def get_perimeter(self) -> float:
        return 2 * math.pi * self.radius

class Triangle(Shape):
    def __init__(self, p1: Point, p2: Point, p3: Point, style: Optional[Style] = None):
        # Üçgenin merkez noktasını hesapla
        center_x = (p1.x + p2.x + p3.x) / 3
        center_y = (p1.y + p2.y + p3.y) / 3
        super().__init__(Point(center_x, center_y), style)
        
        self.points = [p1, p2, p3]
    
    def draw(self) -> str:
        return f"Üçgen çiziliyor: Noktalar {', '.join(str(p) for p in self.points)}"
    
    def resize(self, scale: float) -> None:
        # Merkez noktasına göre tüm noktaları ölçekle
        for point in self.points:
            dx = point.x - self.position.x
            dy = point.y - self.position.y
            point.x = self.position.x + dx * scale
            point.y = self.position.y + dy * scale
    
    def get_area(self) -> float:
        # Üçgenin alanını hesapla (Heron formülü)
        a = math.sqrt((self.points[1].x - self.points[0].x)**2 + 
                     (self.points[1].y - self.points[0].y)**2)
        b = math.sqrt((self.points[2].x - self.points[1].x)**2 + 
                     (self.points[2].y - self.points[1].y)**2)
        c = math.sqrt((self.points[0].x - self.points[2].x)**2 + 
                     (self.points[0].y - self.points[2].y)**2)
        s = (a + b + c) / 2
        return math.sqrt(s * (s - a) * (s - b) * (s - c))
    
    def get_perimeter(self) -> float:
        # Üçgenin çevresini hesapla
        a = math.sqrt((self.points[1].x - self.points[0].x)**2 + 
                     (self.points[1].y - self.points[0].y)**2)
        b = math.sqrt((self.points[2].x - self.points[1].x)**2 + 
                     (self.points[2].y - self.points[1].y)**2)
        c = math.sqrt((self.points[0].x - self.points[2].x)**2 + 
                     (self.points[0].y - self.points[2].y)**2)
        return a + b + c
\`\`\`

### 4. Çizim Yüzeyi

Şekilleri yönetmek ve çizmek için bir yüzey sınıfı:

\`\`\`python
class Canvas:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.shapes: List[Shape] = []
        self.selected_shape: Optional[Shape] = None
    
    def add_shape(self, shape: Shape) -> None:
        self.shapes.append(shape)
    
    def remove_shape(self, shape: Shape) -> None:
        if shape in self.shapes:
            self.shapes.remove(shape)
    
    def select_shape(self, x: float, y: float) -> Optional[Shape]:
        # Basit bir seçim: Verilen noktaya en yakın şekli seç
        min_distance = float('inf')
        selected = None
        
        for shape in self.shapes:
            dx = shape.position.x - x
            dy = shape.position.y - y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                selected = shape
        
        if self.selected_shape:
            self.selected_shape.deselect()
        
        if selected:
            selected.select()
            self.selected_shape = selected
        
        return selected
    
    def move_selected(self, dx: float, dy: float) -> None:
        if self.selected_shape:
            self.selected_shape.move(dx, dy)
    
    def resize_selected(self, scale: float) -> None:
        if self.selected_shape:
            self.selected_shape.resize(scale)
    
    def change_selected_style(self, style: Style) -> None:
        if self.selected_shape:
            self.selected_shape.set_style(style)
    
    def draw_all(self) -> str:
        result = f"Çizim Yüzeyi ({self.width}x{self.height}):\\n"
        for shape in self.shapes:
            result += f"{'[Seçili] ' if shape.selected else ''}{shape.draw()}\\n"
        return result

### 5. Kullanım Örneği

\`\`\`python
# Çizim yüzeyi oluştur
canvas = Canvas(800, 600)

# Şekiller oluştur
rect_style = Style(Color.BLUE, Color.BLACK, LineStyle.SOLID, 2.0)
rectangle = Rectangle(Point(100, 100), 200, 150, rect_style)

circle_style = Style(Color.RED, Color.BLACK, LineStyle.DASHED, 1.5)
circle = Circle(Point(400, 300), 100, circle_style)

triangle_style = Style(Color.YELLOW, Color.BLACK, LineStyle.DOTTED, 1.0)
triangle = Triangle(
    Point(500, 100),
    Point(600, 300),
    Point(400, 300),
    triangle_style
)

# Şekilleri çizim yüzeyine ekle
canvas.add_shape(rectangle)
canvas.add_shape(circle)
canvas.add_shape(triangle)

# Çizim yüzeyini görüntüle
print(canvas.draw_all())

# Bir şekil seç ve değişiklikler yap
selected = canvas.select_shape(400, 300)  # Daireyi seç
if selected:
    print("\\nSeçili şekil bilgisi:")
    print(selected.get_info())
    
    # Şekli taşı
    canvas.move_selected(50, 50)
    
    # Şekli büyüt
    canvas.resize_selected(1.5)
    
    # Stilini değiştir
    new_style = Style(Color.GREEN, Color.BLUE, LineStyle.DASHED, 3.0)
    canvas.change_selected_style(new_style)

# Güncellenmiş çizimi görüntüle
print("\\nGüncellenmiş çizim:")
print(canvas.draw_all())
\`\`\`

## Önemli Noktalar

1. **Soyut Temel Sınıf**: \`Shape\` sınıfı soyut bir temel sınıf olarak tasarlandı ve ortak davranışları tanımladı.

2. **Çok Biçimlilik**: Her şekil kendi \`draw()\`, \`resize()\`, \`get_area()\` ve \`get_perimeter()\` metodlarını özelleştirdi.

3. **Veri Yapıları**: \`Point\`, \`Style\`, \`Color\` ve \`LineStyle\` gibi yardımcı sınıflar ile kod organizasyonu sağlandı.

4. **Şekil Yönetimi**: \`Canvas\` sınıfı ile şekillerin yönetimi ve çizimi merkezi bir yerden yapıldı.

5. **Esnek Tasarım**: Sistem yeni şekil tipleri eklenecek şekilde genişletilebilir.

## Geliştirme Önerileri

1. **Grafik Arayüzü**: Tkinter veya PyQt gibi kütüphaneler ile gerçek bir grafik arayüzü eklenebilir.

2. **Daha Fazla Şekil**: Elips, çokgen gibi yeni şekil tipleri eklenebilir.

3. **Gruplama**: Şekilleri gruplayarak toplu işlem yapma özelliği eklenebilir.

4. **Geri Al/İleri Al**: İşlemleri geri alma ve tekrarlama özelliği eklenebilir.

5. **Dosya İşlemleri**: Çizimleri kaydetme ve yükleme özelliği eklenebilir.
\`\`\`
`; 