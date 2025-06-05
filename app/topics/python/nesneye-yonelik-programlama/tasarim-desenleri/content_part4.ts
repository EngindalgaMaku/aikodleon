export const content_part4 = `
### 4. Iterator (Yineleyici) Deseni

Iterator deseni, bir koleksiyonun iç yapısını açığa çıkarmadan, elemanlarına sırayla erişmeyi sağlar. Bu desen, koleksiyon üzerinde yineleme işlemini koleksiyondan ayırır.

\`\`\`python
from abc import ABC, abstractmethod
from typing import Any, List, Dict

# Iterator: Yineleyici arayüzü
class Iterator(ABC):
    @abstractmethod
    def has_next(self) -> bool:
        pass
    
    @abstractmethod
    def next(self) -> Any:
        pass

# ConcreteIterator: Somut yineleyici
class ArrayIterator(Iterator):
    def __init__(self, collection):
        self._collection = collection
        self._index = 0
    
    def has_next(self) -> bool:
        return self._index < len(self._collection)
    
    def next(self) -> Any:
        if not self.has_next():
            raise StopIteration("Koleksiyonun sonuna ulaşıldı")
        
        value = self._collection[self._index]
        self._index += 1
        return value

class ReverseArrayIterator(Iterator):
    def __init__(self, collection):
        self._collection = collection
        self._index = len(collection) - 1
    
    def has_next(self) -> bool:
        return self._index >= 0
    
    def next(self) -> Any:
        if not self.has_next():
            raise StopIteration("Koleksiyonun başına ulaşıldı")
        
        value = self._collection[self._index]
        self._index -= 1
        return value

# Iterable: Yinelenebilir arayüzü
class Iterable(ABC):
    @abstractmethod
    def create_iterator(self) -> Iterator:
        pass
    
    @abstractmethod
    def create_reverse_iterator(self) -> Iterator:
        pass

# ConcreteIterable: Somut yinelenebilir
class AnimalCollection(Iterable):
    def __init__(self):
        self._animals: List[str] = []
    
    def add(self, animal: str):
        self._animals.append(animal)
    
    def remove(self, animal: str):
        self._animals.remove(animal)
    
    def create_iterator(self) -> Iterator:
        return ArrayIterator(self._animals)
    
    def create_reverse_iterator(self) -> Iterator:
        return ReverseArrayIterator(self._animals)

# Kullanım
def print_all(iterator: Iterator):
    while iterator.has_next():
        print(iterator.next())

# Hayvan koleksiyonu oluştur
animals = AnimalCollection()
animals.add("Kedi")
animals.add("Köpek")
animals.add("Kuş")
animals.add("Balık")

# Normal sırada yinele
print("Normal sırada:")
iterator = animals.create_iterator()
print_all(iterator)

print("\nTers sırada:")
reverse_iterator = animals.create_reverse_iterator()
print_all(reverse_iterator)

# Python'un kendi iterator protokolü
print("\nPython'un kendi iterator protokolü ile:")
class PythonAnimalCollection:
    def __init__(self):
        self._animals = []
    
    def add(self, animal):
        self._animals.append(animal)
    
    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self):
        if self._index < len(self._animals):
            value = self._animals[self._index]
            self._index += 1
            return value
        raise StopIteration

python_animals = PythonAnimalCollection()
python_animals.add("Kedi")
python_animals.add("Köpek")
python_animals.add("Kuş")

for animal in python_animals:
    print(animal)
\`\`\`

### 5. Mediator (Arabulucu) Deseni

Mediator deseni, nesneler arasındaki karmaşık iletişimi kapsülleyerek, nesneler arasındaki doğrudan bağımlılıkları azaltır. Bu desen, nesneler arasındaki iletişimi bir arabulucu nesne üzerinden yürütür.

\`\`\`python
from abc import ABC, abstractmethod
from typing import Dict, List

# Mediator: Arabulucu arayüzü
class ChatMediator(ABC):
    @abstractmethod
    def send_message(self, message: str, user):
        pass
    
    @abstractmethod
    def register_user(self, user):
        pass

# Colleague: Meslektaş arayüzü
class User(ABC):
    def __init__(self, name: str, mediator: ChatMediator):
        self.name = name
        self.mediator = mediator
    
    @abstractmethod
    def send(self, message: str):
        pass
    
    @abstractmethod
    def receive(self, message: str):
        pass

# ConcreteMediator: Somut arabulucu
class ChatRoom(ChatMediator):
    def __init__(self):
        self.users: Dict[str, User] = {}
    
    def register_user(self, user):
        self.users[user.name] = user
        print(f"{user.name} sohbet odasına katıldı")
    
    def send_message(self, message: str, user):
        print(f"[{user.name}]: {message}")
        
        # Mesajı diğer tüm kullanıcılara ilet
        for name, u in self.users.items():
            if u != user:  # Mesajı gönderen hariç
                u.receive(f"{user.name}: {message}")

# ConcreteColleague: Somut meslektaşlar
class ChatUser(User):
    def send(self, message: str):
        print(f"{self.name} mesaj gönderiyor...")
        self.mediator.send_message(message, self)
    
    def receive(self, message: str):
        print(f"{self.name} mesaj aldı: {message}")

class PremiumUser(User):
    def send(self, message: str):
        print(f"Premium kullanıcı {self.name} mesaj gönderiyor...")
        # Premium kullanıcılar mesajlarını öne çıkarabilir
        formatted_message = f"[PREMIUM] {message}"
        self.mediator.send_message(formatted_message, self)
    
    def receive(self, message: str):
        # Premium kullanıcılar bildirimleri özelleştirebilir
        print(f"🔔 {self.name}'e mesaj: {message}")

# Kullanım
chat_room = ChatRoom()

user1 = ChatUser("Ahmet", chat_room)
user2 = ChatUser("Mehmet", chat_room)
user3 = PremiumUser("Ayşe", chat_room)

chat_room.register_user(user1)
chat_room.register_user(user2)
chat_room.register_user(user3)

user1.send("Merhaba herkese!")
user2.send("Merhaba Ahmet, nasılsın?")
user3.send("Herkese selamlar!")
\`\`\`

### 6. Memento (Hatıra) Deseni

Memento deseni, bir nesnenin iç durumunu, daha sonra geri yüklenebilecek şekilde saklamayı sağlar. Bu desen, kapsüllemeyi bozmadan bir nesnenin önceki durumlarına erişmeyi mümkün kılar.

\`\`\`python
from typing import List, Dict, Any
import datetime

# Memento: Hatıra sınıfı
class EditorMemento:
    def __init__(self, content: str, cursor_position: int):
        self._content = content
        self._cursor_position = cursor_position
        self._created_at = datetime.datetime.now()
    
    def get_content(self) -> str:
        return self._content
    
    def get_cursor_position(self) -> int:
        return self._cursor_position
    
    def get_created_at(self) -> datetime.datetime:
        return self._created_at

# Originator: Yaratıcı sınıf
class TextEditor:
    def __init__(self):
        self._content = ""
        self._cursor_position = 0
    
    def type(self, text: str):
        # İmlecin olduğu konuma metin ekle
        self._content = (self._content[:self._cursor_position] + 
                         text + 
                         self._content[self._cursor_position:])
        self._cursor_position += len(text)
    
    def delete(self, count: int = 1):
        # İmlecin olduğu konumdan belirtilen sayıda karakter sil
        if self._cursor_position >= count:
            self._content = (self._content[:self._cursor_position - count] + 
                           self._content[self._cursor_position:])
            self._cursor_position -= count
    
    def move_cursor(self, position: int):
        # İmleci taşı
        if 0 <= position <= len(self._content):
            self._cursor_position = position
    
    def get_content(self) -> str:
        return self._content
    
    def get_cursor_position(self) -> int:
        return self._cursor_position
    
    def create_memento(self) -> EditorMemento:
        # Mevcut durumu bir hatıra olarak kaydet
        return EditorMemento(self._content, self._cursor_position)
    
    def restore(self, memento: EditorMemento):
        # Hatıradan durumu geri yükle
        self._content = memento.get_content()
        self._cursor_position = memento.get_cursor_position()
    
    def __str__(self) -> str:
        # Mevcut durumu görselleştir (imleci göster)
        return (self._content[:self._cursor_position] + 
               "|" + 
               self._content[self._cursor_position:])

# Caretaker: Bakıcı sınıf
class History:
    def __init__(self):
        self._mementos: List[EditorMemento] = []
        self._current_index = -1
    
    def save(self, memento: EditorMemento):
        # Yeni bir değişiklik yapıldığında, o noktadan sonraki geçmiş silinir
        if self._current_index < len(self._mementos) - 1:
            self._mementos = self._mementos[:self._current_index + 1]
        
        self._mementos.append(memento)
        self._current_index = len(self._mementos) - 1
    
    def undo(self) -> EditorMemento:
        if self._current_index <= 0:
            return None
        
        self._current_index -= 1
        return self._mementos[self._current_index]
    
    def redo(self) -> EditorMemento:
        if self._current_index >= len(self._mementos) - 1:
            return None
        
        self._current_index += 1
        return self._mementos[self._current_index]
    
    def get_history(self) -> List[str]:
        # Tüm geçmişi listele
        return [
            f"{i+1}. {memento.get_content()} ({memento.get_created_at().strftime('%H:%M:%S')})"
            for i, memento in enumerate(self._mementos)
        ]

# Kullanım
editor = TextEditor()
history = History()

# İlk durumu kaydet
history.save(editor.create_memento())

# Metin yazma
editor.type("Merhaba ")
print(f"Editör: {editor}")
history.save(editor.create_memento())

editor.type("dünya!")
print(f"Editör: {editor}")
history.save(editor.create_memento())

# İmleç hareketi
editor.move_cursor(7)
print(f"İmleç taşındı: {editor}")
history.save(editor.create_memento())

# Silme
editor.delete(6)
print(f"Silme sonrası: {editor}")
history.save(editor.create_memento())

editor.type("Python")
print(f"Yeni metin: {editor}")
history.save(editor.create_memento())

# Geri al
memento = history.undo()
if memento:
    editor.restore(memento)
    print(f"Geri alındı: {editor}")

memento = history.undo()
if memento:
    editor.restore(memento)
    print(f"Geri alındı: {editor}")

# Yeniden yap
memento = history.redo()
if memento:
    editor.restore(memento)
    print(f"Yeniden yapıldı: {editor}")

# Geçmişi görüntüle
print("\nGeçmiş:")
for entry in history.get_history():
    print(entry)
\`\`\`

### 7. Observer (Gözlemci) Deseni

Observer deseni, bir nesnedeki değişiklikleri diğer nesnelere otomatik olarak bildirir. Bu desen, bir nesneye bağımlı olan diğer nesnelerin, nesne değiştiğinde otomatik olarak güncellenmesini sağlar.

\`\`\`python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Subject: Konu arayüzü
class Subject(ABC):
    @abstractmethod
    def attach(self, observer):
        pass
    
    @abstractmethod
    def detach(self, observer):
        pass
    
    @abstractmethod
    def notify(self):
        pass

# Observer: Gözlemci arayüzü
class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

# ConcreteSubject: Somut konu
class WeatherStation(Subject):
    def __init__(self):
        self._observers: List[Observer] = []
        self._temperature = 0
        self._humidity = 0
        self._pressure = 0
    
    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)
    
    def set_measurements(self, temperature, humidity, pressure):
        self._temperature = temperature
        self._humidity = humidity
        self._pressure = pressure
        self.notify()
    
    def get_temperature(self):
        return self._temperature
    
    def get_humidity(self):
        return self._humidity
    
    def get_pressure(self):
        return self._pressure

# ConcreteObserver: Somut gözlemciler
class CurrentConditionsDisplay(Observer):
    def __init__(self, weather_station):
        self._weather_station = weather_station
        self._weather_station.attach(self)
    
    def update(self, subject):
        if isinstance(subject, WeatherStation):
            self._temperature = subject.get_temperature()
            self._humidity = subject.get_humidity()
            self.display()
    
    def display(self):
        print(f"Güncel Hava Durumu: {self._temperature}°C sıcaklık ve %{self._humidity} nem")

class StatisticsDisplay(Observer):
    def __init__(self, weather_station):
        self._weather_station = weather_station
        self._weather_station.attach(self)
        self._temperature_sum = 0
        self._reading_count = 0
        self._max_temperature = float('-inf')
        self._min_temperature = float('inf')
    
    def update(self, subject):
        if isinstance(subject, WeatherStation):
            temp = subject.get_temperature()
            self._temperature_sum += temp
            self._reading_count += 1
            self._max_temperature = max(self._max_temperature, temp)
            self._min_temperature = min(self._min_temperature, temp)
            self.display()
    
    def display(self):
        avg_temp = self._temperature_sum / self._reading_count if self._reading_count > 0 else 0
        print(f"Sıcaklık İstatistikleri: Ort: {avg_temp:.1f}°C, Min: {self._min_temperature}°C, Max: {self._max_temperature}°C")

class ForecastDisplay(Observer):
    def __init__(self, weather_station):
        self._weather_station = weather_station
        self._weather_station.attach(self)
        self._last_pressure = 0
        self._current_pressure = 0
    
    def update(self, subject):
        if isinstance(subject, WeatherStation):
            self._last_pressure = self._current_pressure
            self._current_pressure = subject.get_pressure()
            self.display()
    
    def display(self):
        forecast = "Tahmin yok"
        if self._last_pressure > 0:
            if self._current_pressure > self._last_pressure:
                forecast = "Hava düzeliyor"
            elif self._current_pressure == self._last_pressure:
                forecast = "Hava aynı kalacak"
            else:
                forecast = "Yağmur bekleniyor"
        
        print(f"Hava Tahmini: {forecast}")

# Kullanım
weather_station = WeatherStation()

current_display = CurrentConditionsDisplay(weather_station)
statistics_display = StatisticsDisplay(weather_station)
forecast_display = ForecastDisplay(weather_station)

print("İlk ölçüm:")
weather_station.set_measurements(27, 65, 1013)

print("\nİkinci ölçüm:")
weather_station.set_measurements(28, 70, 1014)

print("\nÜçüncü ölçüm:")
weather_station.set_measurements(26, 75, 1012)

# Bir gözlemciyi çıkar
print("\nTahmin göstergesini kaldırma:")
weather_station.detach(forecast_display)

print("\nDördüncü ölçüm (tahmin göstergesi olmadan):")
weather_station.set_measurements(25, 80, 1010)
\`\`\`
`; 