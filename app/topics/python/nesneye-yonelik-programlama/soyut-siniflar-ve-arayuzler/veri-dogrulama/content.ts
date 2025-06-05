export const content = `
# Veri Doğrulama Sistemi Alıştırması

Bu alıştırmada, farklı veri türleri için doğrulama işlemleri yapabilen bir sistem tasarlayacağız. Sistem, soyut sınıflar ve protokoller kullanarak veri doğrulama işlevselliğini sağlayacak.

## Problem Tanımı

Aşağıdaki özelliklere sahip bir veri doğrulama sistemi geliştirmeniz gerekiyor:

1. Farklı veri türleri için doğrulama kuralları (metin, sayı, e-posta, telefon, tarih vb.)
2. Özelleştirilebilir hata mesajları
3. Zincirleme doğrulama kuralları
4. Doğrulama sonuçlarının raporlanması
5. Özel doğrulama kuralları ekleme desteği

## Çözüm

\`\`\`python
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Protocol, runtime_checkable
from datetime import datetime
import re

# Doğrulama sonucu için veri sınıfı
class ValidationResult:
    def __init__(self, is_valid: bool, message: Optional[str] = None):
        self.is_valid = is_valid
        self.message = message

# Doğrulayıcı protokolü
@runtime_checkable
class Validator(Protocol):
    def validate(self, value: Any) -> ValidationResult:
        ...

# Temel doğrulayıcı sınıfı
class BaseValidator(ABC):
    def __init__(self, error_message: Optional[str] = None):
        self.error_message = error_message
    
    @abstractmethod
    def validate(self, value: Any) -> ValidationResult:
        pass

# Metin doğrulayıcı
class TextValidator(BaseValidator):
    def __init__(self, min_length: int = 0, max_length: Optional[int] = None, 
                 pattern: Optional[str] = None, error_message: Optional[str] = None):
        super().__init__(error_message)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern and re.compile(pattern)
    
    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(False, "Değer bir metin olmalıdır")
        
        if len(value) < self.min_length:
            return ValidationResult(False, 
                self.error_message or f"Metin en az {self.min_length} karakter olmalıdır")
        
        if self.max_length and len(value) > self.max_length:
            return ValidationResult(False, 
                self.error_message or f"Metin en fazla {self.max_length} karakter olmalıdır")
        
        if self.pattern and not self.pattern.match(value):
            return ValidationResult(False, 
                self.error_message or "Metin belirtilen desene uymalıdır")
        
        return ValidationResult(True)

# Sayı doğrulayıcı
class NumberValidator(BaseValidator):
    def __init__(self, min_value: Optional[float] = None, 
                 max_value: Optional[float] = None,
                 is_integer: bool = False,
                 error_message: Optional[str] = None):
        super().__init__(error_message)
        self.min_value = min_value
        self.max_value = max_value
        self.is_integer = is_integer
    
    def validate(self, value: Any) -> ValidationResult:
        if self.is_integer and not isinstance(value, int):
            return ValidationResult(False, "Değer bir tam sayı olmalıdır")
        
        if not self.is_integer and not isinstance(value, (int, float)):
            return ValidationResult(False, "Değer bir sayı olmalıdır")
        
        if self.min_value is not None and value < self.min_value:
            return ValidationResult(False, 
                self.error_message or f"Değer en az {self.min_value} olmalıdır")
        
        if self.max_value is not None and value > self.max_value:
            return ValidationResult(False, 
                self.error_message or f"Değer en fazla {self.max_value} olmalıdır")
        
        return ValidationResult(True)

# E-posta doğrulayıcı
class EmailValidator(BaseValidator):
    def __init__(self, error_message: Optional[str] = None):
        super().__init__(error_message)
        self.pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(False, "E-posta adresi bir metin olmalıdır")
        
        if not self.pattern.match(value):
            return ValidationResult(False, 
                self.error_message or "Geçersiz e-posta adresi")
        
        return ValidationResult(True)

# Tarih doğrulayıcı
class DateValidator(BaseValidator):
    def __init__(self, min_date: Optional[datetime] = None,
                 max_date: Optional[datetime] = None,
                 format: str = "%Y-%m-%d",
                 error_message: Optional[str] = None):
        super().__init__(error_message)
        self.min_date = min_date
        self.max_date = max_date
        self.format = format
    
    def validate(self, value: Any) -> ValidationResult:
        if isinstance(value, str):
            try:
                value = datetime.strptime(value, self.format)
            except ValueError:
                return ValidationResult(False, 
                    self.error_message or f"Tarih {self.format} formatında olmalıdır")
        
        if not isinstance(value, datetime):
            return ValidationResult(False, "Değer bir tarih olmalıdır")
        
        if self.min_date and value < self.min_date:
            return ValidationResult(False, 
                self.error_message or f"Tarih {self.min_date} tarihinden sonra olmalıdır")
        
        if self.max_date and value > self.max_date:
            return ValidationResult(False, 
                self.error_message or f"Tarih {self.max_date} tarihinden önce olmalıdır")
        
        return ValidationResult(True)

# Zincirleme doğrulayıcı
class ChainValidator(BaseValidator):
    def __init__(self, validators: List[Validator]):
        super().__init__()
        self.validators = validators
    
    def validate(self, value: Any) -> ValidationResult:
        for validator in self.validators:
            result = validator.validate(value)
            if not result.is_valid:
                return result
        return ValidationResult(True)

# Özel doğrulayıcı örneği
class TurkishPhoneValidator(BaseValidator):
    def __init__(self, error_message: Optional[str] = None):
        super().__init__(error_message)
        self.pattern = re.compile(r'^(\+90|0)?[1-9][0-9]{9}$')
    
    def validate(self, value: Any) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(False, "Telefon numarası bir metin olmalıdır")
        
        # Boşlukları kaldır
        value = value.replace(" ", "")
        
        if not self.pattern.match(value):
            return ValidationResult(False, 
                self.error_message or "Geçersiz Türkiye telefon numarası")
        
        return ValidationResult(True)

# Form doğrulama örneği
class UserForm:
    def __init__(self):
        self.validators = {
            'username': ChainValidator([
                TextValidator(min_length=3, max_length=20, 
                            pattern=r'^[a-zA-Z0-9_]+$',
                            error_message="Kullanıcı adı 3-20 karakter arasında olmalı ve sadece harf, rakam ve alt çizgi içermeli"),
            ]),
            'email': EmailValidator(),
            'age': NumberValidator(min_value=18, max_value=120, is_integer=True),
            'phone': TurkishPhoneValidator(),
            'birth_date': DateValidator(
                max_date=datetime.now(),
                error_message="Doğum tarihi bugünden önce olmalıdır"
            )
        }
    
    def validate(self, data: dict) -> dict:
        results = {}
        for field, validator in self.validators.items():
            if field in data:
                results[field] = validator.validate(data[field])
            else:
                results[field] = ValidationResult(False, "Alan gerekli")
        return results

# Kullanım örneği
def main():
    # Form verisi
    user_data = {
        'username': 'john_doe',
        'email': 'john@example.com',
        'age': 25,
        'phone': '+90 555 123 4567',
        'birth_date': '1998-05-15'
    }
    
    # Form doğrulama
    form = UserForm()
    results = form.validate(user_data)
    
    # Sonuçları göster
    print("Form Doğrulama Sonuçları:")
    for field, result in results.items():
        status = "✓" if result.is_valid else "✗"
        message = result.message or "Geçerli"
        print(f"{status} {field}: {message}")
    
    # Özel doğrulama örnekleri
    print("\\nÖzel Doğrulama Örnekleri:")
    
    # Metin doğrulama
    text_validator = TextValidator(min_length=5, max_length=10)
    print("Metin doğrulama:", text_validator.validate("Python").is_valid)  # True
    
    # Sayı doğrulama
    number_validator = NumberValidator(min_value=0, max_value=100)
    print("Sayı doğrulama:", number_validator.validate(42).is_valid)  # True
    
    # E-posta doğrulama
    email_validator = EmailValidator()
    print("E-posta doğrulama:", email_validator.validate("test@example.com").is_valid)  # True
    
    # Telefon doğrulama
    phone_validator = TurkishPhoneValidator()
    print("Telefon doğrulama:", phone_validator.validate("+90 555 123 4567").is_valid)  # True

if __name__ == "__main__":
    main()
\`\`\`

## Önemli Noktlar

1. **Protokol Kullanımı**: \`Validator\` protokolü ile doğrulayıcı arayüzü tanımlanmıştır.
2. **Soyut Temel Sınıf**: \`BaseValidator\` sınıfı, tüm doğrulayıcılar için ortak davranışları tanımlar.
3. **Özelleştirilebilirlik**: Her doğrulayıcı için özel hata mesajları ve parametreler ayarlanabilir.
4. **Zincirleme Doğrulama**: \`ChainValidator\` ile birden fazla doğrulama kuralı birleştirilebilir.
5. **Tip Güvenliği**: Her doğrulayıcı, kendi veri türü için uygun kontroller yapar.

## Geliştirme Önerileri

1. **Asenkron Doğrulama**: Uzak API çağrıları gerektiren doğrulamalar için asenkron destek.
2. **Çoklu Dil Desteği**: Hata mesajları için çoklu dil desteği.
3. **Özelleştirilebilir Raporlama**: JSON, XML gibi farklı formatlarda doğrulama raporları.
4. **Kural Oluşturucu**: Doğrulama kurallarını dinamik olarak oluşturmak için bir arayüz.
5. **Performans Optimizasyonu**: Büyük veri setleri için önbellek ve paralel doğrulama desteği.

Bu örnek, soyut sınıflar ve protokollerin veri doğrulama gibi gerçek dünya problemlerinde nasıl kullanılabileceğini göstermektedir. Sistem, yeni doğrulama türleri eklemek için kolayca genişletilebilir şekilde tasarlanmıştır.
`; 