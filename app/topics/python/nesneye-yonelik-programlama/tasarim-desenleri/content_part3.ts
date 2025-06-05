export const content_part3 = `
## Davranışsal Desenler (Behavioral Patterns)

Davranışsal desenler, nesneler arasındaki sorumluluk dağılımı ve iletişimi ile ilgilenir. Bu desenler, nesneler arasındaki iletişimi düzenleyerek daha esnek bir yapı oluşturur.

### 1. Chain of Responsibility (Sorumluluk Zinciri) Deseni

Chain of Responsibility deseni, bir isteğin işlenme şansını birden fazla nesneye vererek, istekle ilgili nesneler arasında bir zincir oluşturur. Bu desen, isteği gönderen ile işleyenler arasındaki bağlantıyı azaltır.

\`\`\`python
from abc import ABC, abstractmethod

# Handler: İşleyici arayüzü
class LogHandler(ABC):
    def __init__(self):
        self._next_handler = None
    
    def set_next(self, handler):
        self._next_handler = handler
        return handler  # Zincirleme için döndür
    
    def handle(self, log_level, message):
        if self._next_handler:
            return self._next_handler.handle(log_level, message)
        return None

# ConcreteHandlers: Somut işleyiciler
class InfoLogHandler(LogHandler):
    def handle(self, log_level, message):
        if log_level == "INFO":
            return f"INFO: {message}"
        return super().handle(log_level, message)

class WarningLogHandler(LogHandler):
    def handle(self, log_level, message):
        if log_level == "WARNING":
            return f"WARNING: {message}"
        return super().handle(log_level, message)

class ErrorLogHandler(LogHandler):
    def handle(self, log_level, message):
        if log_level == "ERROR":
            return f"ERROR: {message}"
        return super().handle(log_level, message)

class CriticalLogHandler(LogHandler):
    def handle(self, log_level, message):
        if log_level == "CRITICAL":
            return f"CRITICAL: {message} [ACİL MÜDAHALE GEREKLİ]"
        return super().handle(log_level, message)

# Logger: İstemci
class Logger:
    def __init__(self):
        # Zinciri oluştur
        info_handler = InfoLogHandler()
        warning_handler = WarningLogHandler()
        error_handler = ErrorLogHandler()
        critical_handler = CriticalLogHandler()
        
        # Zinciri bağla
        info_handler.set_next(warning_handler).set_next(error_handler).set_next(critical_handler)
        
        self.handler = info_handler
    
    def log(self, level, message):
        result = self.handler.handle(level, message)
        if result:
            print(result)
        else:
            print(f"Log seviyesi '{level}' işlenemedi: {message}")

# Kullanım
logger = Logger()
logger.log("INFO", "Uygulama başlatıldı")
logger.log("WARNING", "Yapılandırma dosyası bulunamadı, varsayılanlar kullanılıyor")
logger.log("ERROR", "Veritabanı bağlantısı başarısız")
logger.log("CRITICAL", "Uygulama çöktü")
logger.log("DEBUG", "Değişken değeri: 42")  # İşlenmeyen log seviyesi
\`\`\`

### 2. Command (Komut) Deseni

Command deseni, bir isteği veya komutu nesne olarak kapsülleyerek, istemcilerin farklı isteklerde bulunmasını, istekleri sıraya koymasını ve geri almasını sağlar. Bu desen, işlemi gerçekleştiren nesneden işlemin parametrelerini ve ayrıntılarını ayırır.

\`\`\`python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Command: Komut arayüzü
class Komut(ABC):
    @abstractmethod
    def calistir(self):
        pass
    
    @abstractmethod
    def geri_al(self):
        pass

# Receiver: Alıcı sınıf
class MetinEditoru:
    def __init__(self):
        self.icerik = ""
        self.secim_baslangic = 0
        self.secim_bitis = 0
    
    def metin_yaz(self, text):
        eski_icerik = self.icerik
        # Seçili alan varsa, seçili metni değiştir
        if self.secim_baslangic != self.secim_bitis:
            self.icerik = (self.icerik[:self.secim_baslangic] + 
                           text + 
                           self.icerik[self.secim_bitis:])
        else:
            # Yoksa, imlecin olduğu yere ekle
            self.icerik = (self.icerik[:self.secim_baslangic] + 
                           text + 
                           self.icerik[self.secim_baslangic:])
        
        # İmleç konumunu güncelle
        self.secim_baslangic += len(text)
        self.secim_bitis = self.secim_baslangic
        
        return eski_icerik
    
    def sil(self):
        if self.secim_baslangic == self.secim_bitis:
            # Seçim yoksa, sonraki karakteri sil
            if self.secim_baslangic < len(self.icerik):
                self.secim_bitis += 1
        
        eski_icerik = self.icerik
        silinen_metin = self.icerik[self.secim_baslangic:self.secim_bitis]
        
        # Seçili alanı sil
        self.icerik = (self.icerik[:self.secim_baslangic] + 
                      self.icerik[self.secim_bitis:])
        
        # İmleç konumunu güncelle
        self.secim_bitis = self.secim_baslangic
        
        return eski_icerik, silinen_metin
    
    def sec(self, baslangic, bitis):
        self.secim_baslangic = max(0, min(baslangic, len(self.icerik)))
        self.secim_bitis = max(0, min(bitis, len(self.icerik)))
    
    def get_durum(self):
        return {
            "icerik": self.icerik,
            "secim_baslangic": self.secim_baslangic,
            "secim_bitis": self.secim_bitis
        }
    
    def set_durum(self, durum):
        self.icerik = durum["icerik"]
        self.secim_baslangic = durum["secim_baslangic"]
        self.secim_bitis = durum["secim_bitis"]

# ConcreteCommand: Somut komut sınıfları
class YazKomutu(Komut):
    def __init__(self, editor: MetinEditoru, metin: str):
        self.editor = editor
        self.metin = metin
        self.eski_durum = None
    
    def calistir(self):
        # Önceki durumu kaydet
        self.eski_durum = self.editor.get_durum()
        self.editor.metin_yaz(self.metin)
    
    def geri_al(self):
        if self.eski_durum:
            self.editor.set_durum(self.eski_durum)

class SilKomutu(Komut):
    def __init__(self, editor: MetinEditoru):
        self.editor = editor
        self.eski_durum = None
    
    def calistir(self):
        # Önceki durumu kaydet
        self.eski_durum = self.editor.get_durum()
        self.editor.sil()
    
    def geri_al(self):
        if self.eski_durum:
            self.editor.set_durum(self.eski_durum)

class SecKomutu(Komut):
    def __init__(self, editor: MetinEditoru, baslangic: int, bitis: int):
        self.editor = editor
        self.baslangic = baslangic
        self.bitis = bitis
        self.eski_secim_baslangic = None
        self.eski_secim_bitis = None
    
    def calistir(self):
        # Önceki seçimi kaydet
        self.eski_secim_baslangic = self.editor.secim_baslangic
        self.eski_secim_bitis = self.editor.secim_bitis
        self.editor.sec(self.baslangic, self.bitis)
    
    def geri_al(self):
        if self.eski_secim_baslangic is not None and self.eski_secim_bitis is not None:
            self.editor.sec(self.eski_secim_baslangic, self.eski_secim_bitis)

# Invoker: Çağırıcı sınıf
class EditorTarihi:
    def __init__(self):
        self.gecmis: List[Komut] = []
        self.geri_alinanlar: List[Komut] = []
    
    def komut_calistir(self, komut: Komut):
        komut.calistir()
        self.gecmis.append(komut)
        self.geri_alinanlar.clear()  # Yeni bir komut çalıştığında geri alınanları temizle
    
    def geri_al(self):
        if not self.gecmis:
            return False
        
        komut = self.gecmis.pop()
        komut.geri_al()
        self.geri_alinanlar.append(komut)
        return True
    
    def yeniden_yap(self):
        if not self.geri_alinanlar:
            return False
        
        komut = self.geri_alinanlar.pop()
        komut.calistir()
        self.gecmis.append(komut)
        return True

# Client: İstemci kodu
class Uygulama:
    def __init__(self):
        self.editor = MetinEditoru()
        self.tarih = EditorTarihi()
    
    def yaz(self, metin):
        komut = YazKomutu(self.editor, metin)
        self.tarih.komut_calistir(komut)
        print(f"Yazıldı: '{metin}'")
        self.durum_goster()
    
    def sil(self):
        komut = SilKomutu(self.editor)
        self.tarih.komut_calistir(komut)
        print("Silindi")
        self.durum_goster()
    
    def sec(self, baslangic, bitis):
        komut = SecKomutu(self.editor, baslangic, bitis)
        self.tarih.komut_calistir(komut)
        print(f"Seçildi: {baslangic}-{bitis}")
        self.durum_goster()
    
    def geri_al(self):
        if self.tarih.geri_al():
            print("Geri alındı")
        else:
            print("Geri alınacak komut yok")
        self.durum_goster()
    
    def yeniden_yap(self):
        if self.tarih.yeniden_yap():
            print("Yeniden yapıldı")
        else:
            print("Yeniden yapılacak komut yok")
        self.durum_goster()
    
    def durum_goster(self):
        durum = self.editor.get_durum()
        icerik = durum["icerik"]
        baslangic = durum["secim_baslangic"]
        bitis = durum["secim_bitis"]
        
        # İçeriği ve seçimi göster
        if baslangic == bitis:
            # İmleç tek bir konumda
            gosterim = icerik[:baslangic] + "|" + icerik[baslangic:]
        else:
            # Metin seçili
            gosterim = icerik[:baslangic] + "[" + icerik[baslangic:bitis] + "]" + icerik[bitis:]
        
        print(f"Metin: {gosterim}")
        print("-" * 40)

# Kullanım
app = Uygulama()
app.yaz("Merhaba ")
app.yaz("Dünya")
app.sec(0, 7)  # "Merhaba" seç
app.yaz("Selam")
app.geri_al()
app.geri_al()
app.yeniden_yap()
app.sil()
app.yaz("Python")
\`\`\`

### 3. Interpreter (Yorumlayıcı) Deseni

Interpreter deseni, belirli bir dildeki ifadeleri yorumlamak için kullanılır. Bu desen, dil ifadelerini temsil eden bir sözdizimi ağacı oluşturarak, her düğümün yorumlanmasını sağlar.

\`\`\`python
from abc import ABC, abstractmethod

# Context: Yorumlayıcının durumunu içerir
class Context:
    def __init__(self):
        self.variables = {}
    
    def get_variable(self, name):
        return self.variables.get(name, 0)
    
    def set_variable(self, name, value):
        self.variables[name] = value

# Expression: Temel ifade arayüzü
class Expression(ABC):
    @abstractmethod
    def interpret(self, context):
        pass

# Terminal Expression: Temel değerleri temsil eden ifadeler
class NumberExpression(Expression):
    def __init__(self, value):
        self.value = value
    
    def interpret(self, context):
        return self.value

class VariableExpression(Expression):
    def __init__(self, name):
        self.name = name
    
    def interpret(self, context):
        return context.get_variable(self.name)

# Non-terminal Expression: İşlemleri temsil eden ifadeler
class AddExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def interpret(self, context):
        return self.left.interpret(context) + self.right.interpret(context)

class SubtractExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def interpret(self, context):
        return self.left.interpret(context) - self.right.interpret(context)

class MultiplyExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def interpret(self, context):
        return self.left.interpret(context) * self.right.interpret(context)

class DivideExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def interpret(self, context):
        right_val = self.right.interpret(context)
        if right_val == 0:
            raise ValueError("Sıfıra bölme hatası")
        return self.left.interpret(context) / right_val

class AssignExpression(Expression):
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression
    
    def interpret(self, context):
        value = self.expression.interpret(context)
        context.set_variable(self.name, value)
        return value

# Basit bir ayrıştırıcı
class Parser:
    def parse(self, expression_str):
        tokens = expression_str.replace("(", " ( ").replace(")", " ) ").split()
        return self.parse_expression(tokens)
    
    def parse_expression(self, tokens):
        if not tokens:
            raise ValueError("Boş ifade")
        
        token = tokens.pop(0)
        
        if token == "(":
            # İfade başlangıcı
            operator = tokens.pop(0)
            
            if operator == "=":
                # Atama işlemi
                variable = tokens.pop(0)
                expression = self.parse_expression(tokens)
                tokens.pop(0)  # Kapanan parantez
                return AssignExpression(variable, expression)
            
            # İki terimli işlem
            left = self.parse_expression(tokens)
            right = self.parse_expression(tokens)
            tokens.pop(0)  # Kapanan parantez
            
            if operator == "+":
                return AddExpression(left, right)
            elif operator == "-":
                return SubtractExpression(left, right)
            elif operator == "*":
                return MultiplyExpression(left, right)
            elif operator == "/":
                return DivideExpression(left, right)
            else:
                raise ValueError(f"Bilinmeyen operatör: {operator}")
        
        # Sayı veya değişken
        if token.isdigit() or (token[0] == "-" and token[1:].isdigit()):
            return NumberExpression(int(token))
        else:
            return VariableExpression(token)

# Kullanım
context = Context()
parser = Parser()

expressions = [
    "(= x 10)",
    "(= y 5)",
    "(+ x y)",
    "(* (- x y) 2)",
    "(= z (/ (* x y) 2))",
    "z"
]

for expr_str in expressions:
    expression = parser.parse(expr_str)
    result = expression.interpret(context)
    print(f"{expr_str} = {result}")

print("\nDeğişkenler:")
for name, value in context.variables.items():
    print(f"{name} = {value}")
\`\`\`
`; 