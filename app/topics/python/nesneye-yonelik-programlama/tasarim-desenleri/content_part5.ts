export const content_part5 = `
### 8. State (Durum) Deseni

State deseni, bir nesnenin iç durumu değiştiğinde davranışını değiştirmesini sağlar. Bu desen, duruma bağlı davranışları ayrı durum nesnelerine dönüştürür.

\`\`\`python
from abc import ABC, abstractmethod

# State: Durum arayüzü
class ATMState(ABC):
    @abstractmethod
    def insert_card(self, atm):
        pass
    
    @abstractmethod
    def eject_card(self, atm):
        pass
    
    @abstractmethod
    def enter_pin(self, atm, pin):
        pass
    
    @abstractmethod
    def withdraw_cash(self, atm, amount):
        pass

# Concrete States: Somut durum sınıfları
class NoCardState(ATMState):
    def insert_card(self, atm):
        print("Kart takıldı")
        atm.set_state(atm.get_has_card_state())
    
    def eject_card(self, atm):
        print("Takılı kart yok")
    
    def enter_pin(self, atm, pin):
        print("Önce kart takın")
    
    def withdraw_cash(self, atm, amount):
        print("Önce kart takın")

class HasCardState(ATMState):
    def insert_card(self, atm):
        print("Zaten bir kart takılı")
    
    def eject_card(self, atm):
        print("Kart çıkarıldı")
        atm.set_state(atm.get_no_card_state())
    
    def enter_pin(self, atm, pin):
        if pin == atm.get_correct_pin():
            print("PIN doğru")
            atm.set_state(atm.get_pin_correct_state())
        else:
            print("PIN yanlış")
            print("Kart çıkarıldı")
            atm.set_state(atm.get_no_card_state())
    
    def withdraw_cash(self, atm, amount):
        print("Önce PIN girin")

class PinCorrectState(ATMState):
    def insert_card(self, atm):
        print("Zaten bir kart takılı")
    
    def eject_card(self, atm):
        print("Kart çıkarıldı")
        atm.set_state(atm.get_no_card_state())
    
    def enter_pin(self, atm, pin):
        print("PIN zaten girildi")
    
    def withdraw_cash(self, atm, amount):
        if amount <= atm.get_cash_in_machine():
            atm.reduce_cash_in_machine(amount)
            print(f"{amount} TL çekildi")
            print("Kart çıkarıldı")
            atm.set_state(atm.get_no_card_state())
        else:
            print("Yetersiz bakiye")
            print("Kart çıkarıldı")
            atm.set_state(atm.get_no_card_state())

# Context: Bağlam sınıfı
class ATMMachine:
    def __init__(self, cash_in_machine):
        # Durumlar
        self._no_card_state = NoCardState()
        self._has_card_state = HasCardState()
        self._pin_correct_state = PinCorrectState()
        
        # Mevcut durum
        self._state = self._no_card_state
        
        # ATM'nin içindeki para
        self._cash_in_machine = cash_in_machine
        
        # Doğru PIN (gerçekte kart nesnesinde olur)
        self._correct_pin = "1234"
    
    # Durum değiştirme metodu
    def set_state(self, state):
        self._state = state
    
    # Durum erişim metodları
    def get_no_card_state(self):
        return self._no_card_state
    
    def get_has_card_state(self):
        return self._has_card_state
    
    def get_pin_correct_state(self):
        return self._pin_correct_state
    
    # ATM işlem metodları (bunlar duruma yönlendirilir)
    def insert_card(self):
        self._state.insert_card(self)
    
    def eject_card(self):
        self._state.eject_card(self)
    
    def enter_pin(self, pin):
        self._state.enter_pin(self, pin)
    
    def withdraw_cash(self, amount):
        self._state.withdraw_cash(self, amount)
    
    # Yardımcı metodlar
    def get_cash_in_machine(self):
        return self._cash_in_machine
    
    def reduce_cash_in_machine(self, amount):
        self._cash_in_machine -= amount
    
    def get_correct_pin(self):
        return self._correct_pin

# Kullanım
atm = ATMMachine(2000)

print("ATM'de işlem sırası:")
atm.withdraw_cash(100)  # Kart takılı değil
atm.insert_card()       # Kart takıldı
atm.withdraw_cash(100)  # PIN girilmedi
atm.enter_pin("1111")   # Yanlış PIN
atm.insert_card()       # Tekrar kart takıldı
atm.enter_pin("1234")   # Doğru PIN
atm.withdraw_cash(1500) # Para çekildi
atm.insert_card()       # Yeni işlem
atm.enter_pin("1234")   # PIN doğru
atm.withdraw_cash(700)  # Kalan para çekildi
atm.withdraw_cash(100)  # Yetersiz bakiye
\`\`\`

### 9. Strategy (Strateji) Deseni

Strategy deseni, bir algoritma ailesini tanımlayıp, her birini kapsülleyerek, birbirlerinin yerine kullanılabilir hale getirir. Bu desen, algoritmaları kullanıcılarından bağımsız olarak değiştirmeyi mümkün kılar.

\`\`\`python
from abc import ABC, abstractmethod
from typing import List

# Strategy: Strateji arayüzü
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List) -> List:
        pass

# ConcreteStrategy: Somut strateji sınıfları
class BubbleSortStrategy(SortStrategy):
    def sort(self, data: List) -> List:
        print("Kabarcık sıralaması uygulanıyor...")
        result = data.copy()
        n = len(result)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
        
        return result

class QuickSortStrategy(SortStrategy):
    def sort(self, data: List) -> List:
        print("Hızlı sıralama uygulanıyor...")
        result = data.copy()
        self._quick_sort(result, 0, len(result) - 1)
        return result
    
    def _quick_sort(self, data, low, high):
        if low < high:
            pi = self._partition(data, low, high)
            self._quick_sort(data, low, pi - 1)
            self._quick_sort(data, pi + 1, high)
    
    def _partition(self, data, low, high):
        pivot = data[high]
        i = low - 1
        
        for j in range(low, high):
            if data[j] <= pivot:
                i += 1
                data[i], data[j] = data[j], data[i]
        
        data[i + 1], data[high] = data[high], data[i + 1]
        return i + 1

class MergeSortStrategy(SortStrategy):
    def sort(self, data: List) -> List:
        print("Birleştirme sıralaması uygulanıyor...")
        if len(data) <= 1:
            return data
        
        result = data.copy()
        self._merge_sort(result, 0, len(result) - 1)
        return result
    
    def _merge_sort(self, data, left, right):
        if left < right:
            middle = (left + right) // 2
            self._merge_sort(data, left, middle)
            self._merge_sort(data, middle + 1, right)
            self._merge(data, left, middle, right)
    
    def _merge(self, data, left, middle, right):
        n1 = middle - left + 1
        n2 = right - middle
        
        L = [0] * n1
        R = [0] * n2
        
        for i in range(n1):
            L[i] = data[left + i]
        
        for j in range(n2):
            R[j] = data[middle + 1 + j]
        
        i = 0
        j = 0
        k = left
        
        while i < n1 and j < n2:
            if L[i] <= R[j]:
                data[k] = L[i]
                i += 1
            else:
                data[k] = R[j]
                j += 1
            k += 1
        
        while i < n1:
            data[k] = L[i]
            i += 1
            k += 1
        
        while j < n2:
            data[k] = R[j]
            j += 1
            k += 1

# Context: Bağlam sınıfı
class Sorter:
    def __init__(self, strategy: SortStrategy = None):
        self._strategy = strategy or BubbleSortStrategy()
    
    def set_strategy(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def sort(self, data: List) -> List:
        return self._strategy.sort(data)

# Kullanım
def print_result(data, sorted_data):
    print(f"Orijinal: {data}")
    print(f"Sıralanmış: {sorted_data}")
    print("-" * 40)

# Veri ve sıralayıcı oluştur
data = [8, 5, 2, 9, 1, 6, 3, 7, 4]
sorter = Sorter()

# Farklı stratejilerle sırala
sorter.set_strategy(BubbleSortStrategy())
bubble_result = sorter.sort(data)
print_result(data, bubble_result)

sorter.set_strategy(QuickSortStrategy())
quick_result = sorter.sort(data)
print_result(data, quick_result)

sorter.set_strategy(MergeSortStrategy())
merge_result = sorter.sort(data)
print_result(data, merge_result)
\`\`\`

### 10. Template Method (Şablon Metodu) Deseni

Template Method deseni, bir algoritmanın iskeletini tanımlayıp, bazı adımlarını alt sınıflara bırakır. Bu desen, algoritmanın yapısını değiştirmeden, belirli adımlarının alt sınıflar tarafından yeniden tanımlanmasını sağlar.

\`\`\`python
from abc import ABC, abstractmethod

# AbstractClass: Soyut sınıf
class DataMiner(ABC):
    # Template method
    def mine_data(self, path):
        file = self.open_file(path)
        data = self.extract_data(file)
        analysis = self.analyze_data(data)
        self.send_report(analysis)
        self.close_file(file)
    
    @abstractmethod
    def open_file(self, path):
        pass
    
    @abstractmethod
    def extract_data(self, file):
        pass
    
    @abstractmethod
    def analyze_data(self, data):
        pass
    
    def send_report(self, analysis):
        print(f"Rapor gönderiliyor: {analysis}")
    
    def close_file(self, file):
        print(f"Dosya kapatılıyor: {file}")

# ConcreteClass: Somut sınıflar
class PDFDataMiner(DataMiner):
    def open_file(self, path):
        print(f"PDF dosyası açılıyor: {path}")
        return f"PDF_{path}"
    
    def extract_data(self, file):
        print(f"{file} dosyasından PDF verileri çıkarılıyor")
        return ["PDF Veri 1", "PDF Veri 2"]
    
    def analyze_data(self, data):
        print("PDF verileri analiz ediliyor")
        return f"PDF Analiz Sonucu: {len(data)} öğe işlendi"

class CSVDataMiner(DataMiner):
    def open_file(self, path):
        print(f"CSV dosyası açılıyor: {path}")
        return f"CSV_{path}"
    
    def extract_data(self, file):
        print(f"{file} dosyasından CSV verileri çıkarılıyor")
        return ["CSV Satır 1", "CSV Satır 2", "CSV Satır 3"]
    
    def analyze_data(self, data):
        print("CSV verileri analiz ediliyor")
        return f"CSV Analiz Sonucu: {len(data)} satır işlendi"

class DatabaseDataMiner(DataMiner):
    def open_file(self, path):
        print(f"Veritabanına bağlanılıyor: {path}")
        return f"DB_{path}"
    
    def extract_data(self, file):
        print(f"{file} veritabanından veriler sorgalanıyor")
        return {"tablo1": 10, "tablo2": 20}
    
    def analyze_data(self, data):
        print("Veritabanı verileri analiz ediliyor")
        total = sum(data.values())
        return f"Veritabanı Analiz Sonucu: {len(data)} tablo, toplam {total} kayıt işlendi"
    
    # İsteğe bağlı bir adımı override et
    def send_report(self, analysis):
        print(f"Veritabanı raporu yöneticiye e-posta olarak gönderiliyor: {analysis}")

# Kullanım
def process_data(data_miner, path):
    print(f"\n{data_miner.__class__.__name__} işlemi başlıyor...")
    data_miner.mine_data(path)
    print("İşlem tamamlandı\n")

pdf_miner = PDFDataMiner()
csv_miner = CSVDataMiner()
db_miner = DatabaseDataMiner()

process_data(pdf_miner, "rapor.pdf")
process_data(csv_miner, "veri.csv")
process_data(db_miner, "uygulama_db")
\`\`\`

### 11. Visitor (Ziyaretçi) Deseni

Visitor deseni, bir nesne yapısındaki elemanlara uygulanacak işlemleri ayırarak, yeni işlemlerin tanımlanmasını kolaylaştırır. Bu desen, sınıf hiyerarşisini değiştirmeden, yeni işlevler eklemeyi mümkün kılar.

\`\`\`python
from abc import ABC, abstractmethod
from typing import List

# Element: Ziyaret edilecek arayüz
class Shape(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass

# ConcreteElements: Ziyaret edilecek somut sınıflar
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def accept(self, visitor):
        return visitor.visit_circle(self)

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def accept(self, visitor):
        return visitor.visit_rectangle(self)

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height
    
    def accept(self, visitor):
        return visitor.visit_triangle(self)

# Visitor: Ziyaretçi arayüzü
class ShapeVisitor(ABC):
    @abstractmethod
    def visit_circle(self, circle):
        pass
    
    @abstractmethod
    def visit_rectangle(self, rectangle):
        pass
    
    @abstractmethod
    def visit_triangle(self, triangle):
        pass

# ConcreteVisitor: Somut ziyaretçi sınıfları
class AreaCalculator(ShapeVisitor):
    def visit_circle(self, circle):
        return 3.14 * circle.radius ** 2
    
    def visit_rectangle(self, rectangle):
        return rectangle.width * rectangle.height
    
    def visit_triangle(self, triangle):
        return 0.5 * triangle.base * triangle.height

class PerimeterCalculator(ShapeVisitor):
    def visit_circle(self, circle):
        return 2 * 3.14 * circle.radius
    
    def visit_rectangle(self, rectangle):
        return 2 * (rectangle.width + rectangle.height)
    
    def visit_triangle(self, triangle):
        # Basitleştirilmiş hesaplama (eşkenar üçgen varsayımı)
        return 3 * triangle.base

class ShapeDrawer(ShapeVisitor):
    def visit_circle(self, circle):
        return f"Daire çiziliyor (yarıçap: {circle.radius})"
    
    def visit_rectangle(self, rectangle):
        return f"Dikdörtgen çiziliyor (genişlik: {rectangle.width}, yükseklik: {rectangle.height})"
    
    def visit_triangle(self, triangle):
        return f"Üçgen çiziliyor (taban: {triangle.base}, yükseklik: {triangle.height})"

# Object Structure: Nesne yapısı
class Drawing:
    def __init__(self):
        self.shapes: List[Shape] = []
    
    def add(self, shape):
        self.shapes.append(shape)
    
    def remove(self, shape):
        self.shapes.remove(shape)
    
    def accept(self, visitor):
        results = []
        for shape in self.shapes:
            results.append(shape.accept(visitor))
        return results

# Kullanım
# Şekiller oluştur
circle = Circle(5)
rectangle = Rectangle(4, 6)
triangle = Triangle(3, 4)

# Çizim oluştur ve şekilleri ekle
drawing = Drawing()
drawing.add(circle)
drawing.add(rectangle)
drawing.add(triangle)

# Ziyaretçiler oluştur
area_calculator = AreaCalculator()
perimeter_calculator = PerimeterCalculator()
drawer = ShapeDrawer()

# Ziyaretçileri kullan
areas = drawing.accept(area_calculator)
perimeters = drawing.accept(perimeter_calculator)
drawings = drawing.accept(drawer)

# Sonuçları göster
print("Alanlar:")
for i, area in enumerate(areas):
    print(f"Şekil {i+1}: {area:.2f}")

print("\nÇevreler:")
for i, perimeter in enumerate(perimeters):
    print(f"Şekil {i+1}: {perimeter:.2f}")

print("\nÇizimler:")
for drawing in drawings:
    print(drawing)
\`\`\`

## Alıştırmalar

1. [Restoran Sipariş Sistemi](/topics/python/nesneye-yonelik-programlama/tasarim-desenleri/restoran-siparis-sistemi)
   - Factory, Strategy ve Observer desenlerini kullanarak bir restoran sipariş sistemi geliştirin
   - Farklı ödeme yöntemleri (Strateji)
   - Sipariş oluşturma (Fabrika)
   - Sipariş durumu bildirimleri (Gözlemci)

2. [Akıllı Ev Sistemi](/topics/python/nesneye-yonelik-programlama/tasarim-desenleri/akilli-ev-sistemi)
   - Command, Facade ve Singleton desenlerini kullanarak bir akıllı ev sistemi uygulaması geliştirin
   - Cihaz kontrolü için komutlar
   - Karmaşık işlemler için ön yüz
   - Merkezi sistem kontrolü için tekil nesne

3. [Metin Editörü](/topics/python/nesneye-yonelik-programlama/tasarim-desenleri/metin-editoru)
   - Memento, Command ve Composite desenlerini kullanarak basit bir metin editörü uygulaması geliştirin
   - Metin içeriğini düzenlemek için komutlar
   - Geri alma/yeniden yapma için anılar
   - Belge yapısı için bileşik desen

## Sonuç

Tasarım desenleri, yazılım geliştirme sürecinde karşılaşılan yaygın problemlere yönelik test edilmiş ve kanıtlanmış çözüm şablonlarıdır. Bu desenler, kodun okunabilirliğini, sürdürülebilirliğini ve esnekliğini artırır.

Bu bölümde, üç ana kategorideki tasarım desenlerini inceledik:

1. **Yaratımsal Desenler**: Nesne oluşturma mekanizmalarıyla ilgilenir.
2. **Yapısal Desenler**: Sınıfların ve nesnelerin bir araya getirilmesiyle ilgilenir.
3. **Davranışsal Desenler**: Nesneler arasındaki iletişim ve sorumluluk dağılımıyla ilgilenir.

Tasarım desenleri, her probleme uygulanabilecek sihirli çözümler değildir. Her desenin avantajları ve dezavantajları vardır. Önemli olan, projenizin gereksinimlerini ve bağlamını anlamak ve uygun deseni seçmektir.

::: tip
"Tasarım desenleri kullanmak için kullanmayın, gerçekten ihtiyacınız olduğunda kullanın."
:::

Bir sonraki adım olarak, bu desenleri gerçek dünya projelerinde nasıl uygulayacağınızı keşfetmek ve desenler arasındaki ilişkileri anlamak olabilir.
`; 