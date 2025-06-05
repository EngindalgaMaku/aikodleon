export const content = `
# Oyun Karakter Sistemi

Bu alıştırmada, çok biçimliliği kullanarak bir RPG oyunu için karakter sistemi geliştireceğiz.

## Problem Tanımı

Farklı karakter sınıflarını (savaşçı, büyücü, okçu vb.) destekleyen bir RPG karakter sistemi geliştirmemiz gerekiyor. Sistem şu özelliklere sahip olmalı:

* Farklı karakter sınıfları için ortak bir arayüz
* Her karakter sınıfı için özel yetenekler ve davranışlar
* Seviye atlama ve ekipman sistemi
* Karakter özellikleri ve istatistikler

## Çözüm

### 1. Temel Veri Yapıları

Önce gerekli veri yapılarını tanımlayalım:

\`\`\`python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional

# Karakter özellikleri
class Stat(Enum):
    STRENGTH = "güç"        # Fiziksel hasar ve taşıma kapasitesi
    DEXTERITY = "çeviklik"  # Hız ve hassasiyet
    INTELLIGENCE = "zeka"   # Büyü gücü ve mana
    VITALITY = "dayanıklılık"  # Can puanı ve savunma
    WISDOM = "bilgelik"     # Mana yenilenmesi ve büyü direnci

# Ekipman türleri
class EquipmentType(Enum):
    WEAPON = "silah"
    ARMOR = "zırh"
    ACCESSORY = "aksesuar"

# Hasar türleri
class DamageType(Enum):
    PHYSICAL = "fiziksel"
    MAGICAL = "büyüsel"
    TRUE = "gerçek"  # Savunmayı görmezden gelen hasar

# Ekipman
@dataclass
class Equipment:
    name: str
    type: EquipmentType
    level_req: int
    stats: Dict[Stat, int]
    
    def __str__(self) -> str:
        stats_str = ", ".join(f"{stat.value}: {value}" for stat, value in self.stats.items())
        return f"{self.name} (Seviye {self.level_req}) [{stats_str}]"

# Yetenek
@dataclass
class Ability:
    name: str
    damage: float
    damage_type: DamageType
    mana_cost: int
    cooldown: int
    description: str
    
    def __str__(self) -> str:
        return f"{self.name} - {self.damage} {self.damage_type.value} hasar (Mana: {self.mana_cost}, Bekleme: {self.cooldown}s)"
\`\`\`

### 2. Temel Karakter Sınıfı

Tüm karakterler için temel bir soyut sınıf oluşturalım:

\`\`\`python
class Character(ABC):
    def __init__(self, name: str):
        self.name = name
        self.level = 1
        self.experience = 0
        self.base_stats = {stat: 10 for stat in Stat}
        self.equipment: Dict[EquipmentType, Optional[Equipment]] = {
            type: None for type in EquipmentType
        }
        self.abilities: List[Ability] = []
        self.health = 100
        self.mana = 100
    
    @abstractmethod
    def level_up(self) -> None:
        pass
    
    @abstractmethod
    def attack(self, target: 'Character') -> str:
        pass
    
    @abstractmethod
    def use_ability(self, ability: Ability, target: 'Character') -> str:
        pass
    
    def get_total_stats(self) -> Dict[Stat, int]:
        # Temel özellikler + ekipman bonusları
        total_stats = self.base_stats.copy()
        
        for equipment in self.equipment.values():
            if equipment:
                for stat, value in equipment.stats.items():
                    total_stats[stat] += value
        
        return total_stats
    
    def equip(self, item: Equipment) -> str:
        if item.level_req > self.level:
            return f"{item.name} için gereken seviyeye ulaşılmadı"
        
        self.equipment[item.type] = item
        return f"{item.name} kuşanıldı"
    
    def unequip(self, type: EquipmentType) -> Optional[Equipment]:
        item = self.equipment[type]
        if item:
            self.equipment[type] = None
            return item
        return None
    
    def gain_experience(self, amount: int) -> str:
        self.experience += amount
        
        # Her 100 deneyim puanında seviye atla
        if self.experience >= self.level * 100:
            self.experience -= self.level * 100
            self.level_up()
            return f"{self.name} seviye atladı! Yeni seviye: {self.level}"
        
        return f"{self.name} {amount} deneyim puanı kazandı"
    
    def get_info(self) -> str:
        stats = self.get_total_stats()
        equipment_str = "\\n".join(
            f"{type.value}: {item or 'Boş'}"
            for type, item in self.equipment.items()
        )
        abilities_str = "\\n".join(str(ability) for ability in self.abilities)
        
        return f"""
{self.name} (Seviye {self.level})
Deneyim: {self.experience}/{self.level * 100}
Can: {self.health}/100
Mana: {self.mana}/100

Özellikler:
{chr(10).join(f'{stat.value}: {value}' for stat, value in stats.items())}

Ekipmanlar:
{equipment_str}

Yetenekler:
{abilities_str}
"""
\`\`\`

### 3. Özel Karakter Sınıfları

Farklı karakter sınıfları için özel sınıflar oluşturalım:

\`\`\`python
class Warrior(Character):
    def __init__(self, name: str):
        super().__init__(name)
        # Savaşçı başlangıç özellikleri
        self.base_stats[Stat.STRENGTH] = 15
        self.base_stats[Stat.VITALITY] = 15
        
        # Savaşçı yetenekleri
        self.abilities = [
            Ability("Güçlü Vuruş", 30, DamageType.PHYSICAL, 20, 5,
                   "Güçlü bir fiziksel saldırı"),
            Ability("Savunma Duruşu", 0, DamageType.PHYSICAL, 15, 10,
                   "Savunmayı artırır")
        ]
    
    def level_up(self) -> None:
        self.level += 1
        self.base_stats[Stat.STRENGTH] += 3
        self.base_stats[Stat.VITALITY] += 3
        self.base_stats[Stat.DEXTERITY] += 1
        self.health = 100
        self.mana = 100
    
    def attack(self, target: Character) -> str:
        damage = self.get_total_stats()[Stat.STRENGTH] * 1.5
        return f"{self.name} {target.name}'e {damage:.1f} fiziksel hasar verdi"
    
    def use_ability(self, ability: Ability, target: Character) -> str:
        if ability not in self.abilities:
            return f"{self.name} bu yeteneğe sahip değil"
        
        if self.mana < ability.mana_cost:
            return f"{self.name}'in manası yetersiz"
        
        self.mana -= ability.mana_cost
        
        if ability.name == "Güçlü Vuruş":
            damage = ability.damage * (1 + self.get_total_stats()[Stat.STRENGTH] / 50)
            return f"{self.name} Güçlü Vuruş kullanarak {target.name}'e {damage:.1f} hasar verdi"
        elif ability.name == "Savunma Duruşu":
            self.base_stats[Stat.VITALITY] += 5  # Geçici savunma artışı
            return f"{self.name} savunma duruşuna geçti"
        
        return "Bilinmeyen yetenek"

class Mage(Character):
    def __init__(self, name: str):
        super().__init__(name)
        # Büyücü başlangıç özellikleri
        self.base_stats[Stat.INTELLIGENCE] = 15
        self.base_stats[Stat.WISDOM] = 15
        
        # Büyücü yetenekleri
        self.abilities = [
            Ability("Ateş Topu", 40, DamageType.MAGICAL, 30, 3,
                   "Güçlü bir büyü saldırısı"),
            Ability("Buz Kalkanı", 0, DamageType.MAGICAL, 25, 8,
                   "Büyü hasarına karşı koruma sağlar")
        ]
    
    def level_up(self) -> None:
        self.level += 1
        self.base_stats[Stat.INTELLIGENCE] += 3
        self.base_stats[Stat.WISDOM] += 3
        self.base_stats[Stat.VITALITY] += 1
        self.health = 100
        self.mana = 100
    
    def attack(self, target: Character) -> str:
        damage = self.get_total_stats()[Stat.INTELLIGENCE]
        return f"{self.name} {target.name}'e {damage:.1f} büyü hasarı verdi"
    
    def use_ability(self, ability: Ability, target: Character) -> str:
        if ability not in self.abilities:
            return f"{self.name} bu yeteneğe sahip değil"
        
        if self.mana < ability.mana_cost:
            return f"{self.name}'in manası yetersiz"
        
        self.mana -= ability.mana_cost
        
        if ability.name == "Ateş Topu":
            damage = ability.damage * (1 + self.get_total_stats()[Stat.INTELLIGENCE] / 50)
            return f"{self.name} Ateş Topu kullanarak {target.name}'e {damage:.1f} büyü hasarı verdi"
        elif ability.name == "Buz Kalkanı":
            self.base_stats[Stat.WISDOM] += 5  # Geçici büyü direnci artışı
            return f"{self.name} buz kalkanı oluşturdu"
        
        return "Bilinmeyen yetenek"

class Archer(Character):
    def __init__(self, name: str):
        super().__init__(name)
        # Okçu başlangıç özellikleri
        self.base_stats[Stat.DEXTERITY] = 15
        self.base_stats[Stat.STRENGTH] = 12
        
        # Okçu yetenekleri
        self.abilities = [
            Ability("Çoklu Atış", 25, DamageType.PHYSICAL, 25, 4,
                   "Birden fazla ok atışı"),
            Ability("Nişancı Gözü", 0, DamageType.PHYSICAL, 20, 6,
                   "Kritik vuruş şansını artırır")
        ]
    
    def level_up(self) -> None:
        self.level += 1
        self.base_stats[Stat.DEXTERITY] += 3
        self.base_stats[Stat.STRENGTH] += 2
        self.base_stats[Stat.VITALITY] += 1
        self.health = 100
        self.mana = 100
    
    def attack(self, target: Character) -> str:
        damage = (self.get_total_stats()[Stat.DEXTERITY] * 0.8 +
                 self.get_total_stats()[Stat.STRENGTH] * 0.4)
        return f"{self.name} {target.name}'e {damage:.1f} fiziksel hasar verdi"
    
    def use_ability(self, ability: Ability, target: Character) -> str:
        if ability not in self.abilities:
            return f"{self.name} bu yeteneğe sahip değil"
        
        if self.mana < ability.mana_cost:
            return f"{self.name}'in manası yetersiz"
        
        self.mana -= ability.mana_cost
        
        if ability.name == "Çoklu Atış":
            damage = ability.damage * (1 + self.get_total_stats()[Stat.DEXTERITY] / 50)
            hits = 3  # 3 ok at
            total_damage = damage * hits
            return f"{self.name} Çoklu Atış kullanarak {target.name}'e {hits} ok ile toplam {total_damage:.1f} hasar verdi"
        elif ability.name == "Nişancı Gözü":
            self.base_stats[Stat.DEXTERITY] += 5  # Geçici çeviklik artışı
            return f"{self.name} nişancı gözünü aktifleştirdi"
        
        return "Bilinmeyen yetenek"
\`\`\`

### 4. Oyun Yöneticisi

Karakterleri ve savaş sistemini yönetmek için bir sınıf:

\`\`\`python
class GameManager:
    def __init__(self):
        self.characters: List[Character] = []
    
    def create_character(self, name: str, character_class: str) -> Character:
        if character_class.lower() == "warrior":
            character = Warrior(name)
        elif character_class.lower() == "mage":
            character = Mage(name)
        elif character_class.lower() == "archer":
            character = Archer(name)
        else:
            raise ValueError("Geçersiz karakter sınıfı")
        
        self.characters.append(character)
        return character
    
    def create_equipment(self, name: str, type: EquipmentType, level_req: int,
                        stats: Dict[Stat, int]) -> Equipment:
        return Equipment(name, type, level_req, stats)
    
    def battle(self, char1: Character, char2: Character, rounds: int = 3) -> str:
        result = f"\\n{char1.name} vs {char2.name}\\n"
        
        for round in range(1, rounds + 1):
            result += f"\\nRound {round}:\\n"
            
            # İlk karakter saldırır
            if char1.abilities and char1.mana >= char1.abilities[0].mana_cost:
                result += char1.use_ability(char1.abilities[0], char2) + "\\n"
            else:
                result += char1.attack(char2) + "\\n"
            
            # İkinci karakter saldırır
            if char2.abilities and char2.mana >= char2.abilities[0].mana_cost:
                result += char2.use_ability(char2.abilities[0], char1) + "\\n"
            else:
                result += char2.attack(char1) + "\\n"
        
        return result

### 5. Kullanım Örneği

\`\`\`python
# Oyun yöneticisi oluştur
game = GameManager()

# Karakterler oluştur
warrior = game.create_character("Aragorn", "warrior")
mage = game.create_character("Gandalf", "mage")
archer = game.create_character("Legolas", "archer")

# Ekipmanlar oluştur
sword = game.create_equipment(
    "Ejder Kılıcı",
    EquipmentType.WEAPON,
    5,
    {Stat.STRENGTH: 10, Stat.VITALITY: 5}
)

staff = game.create_equipment(
    "Bilge Asası",
    EquipmentType.WEAPON,
    5,
    {Stat.INTELLIGENCE: 10, Stat.WISDOM: 5}
)

bow = game.create_equipment(
    "Rüzgar Yayı",
    EquipmentType.WEAPON,
    5,
    {Stat.DEXTERITY: 10, Stat.STRENGTH: 5}
)

# Ekipmanları kuşan
print(warrior.equip(sword))
print(mage.equip(staff))
print(archer.equip(bow))

# Karakter bilgilerini görüntüle
print("\\nKarakter Bilgileri:")
print(warrior.get_info())
print(mage.get_info())
print(archer.get_info())

# Savaş simülasyonu
print("\\nSavaş Simülasyonu:")
print(game.battle(warrior, mage))
print(game.battle(archer, warrior))

# Deneyim kazan ve seviye atla
print("\\nDeneyim ve Seviye Atlama:")
print(warrior.gain_experience(100))
print(warrior.get_info())
\`\`\`

## Önemli Noktalar

1. **Soyut Temel Sınıf**: \`Character\` sınıfı soyut bir temel sınıf olarak tasarlandı ve ortak davranışları tanımladı.

2. **Çok Biçimlilik**: Her karakter sınıfı kendi \`level_up()\`, \`attack()\` ve \`use_ability()\` metodlarını özelleştirdi.

3. **Veri Yapıları**: \`Equipment\`, \`Ability\` ve çeşitli enum sınıfları ile kod organizasyonu sağlandı.

4. **Karakter Yönetimi**: \`GameManager\` sınıfı ile karakter oluşturma ve savaş sistemi merkezi bir yerden yönetildi.

5. **Esnek Tasarım**: Sistem yeni karakter sınıfları ve yetenekler eklenecek şekilde genişletilebilir.

## Geliştirme Önerileri

1. **Envanter Sistemi**: Karakterlerin birden fazla ekipman taşıyabilmesi için envanter sistemi eklenebilir.

2. **Yetenek Ağacı**: Karakterlerin seviye atladıkça yeni yetenekler öğrenebileceği bir sistem eklenebilir.

3. **Etkileşimler**: Karakterler arası ticaret, grup oluşturma gibi sosyal özellikler eklenebilir.

4. **Görev Sistemi**: Karakterlerin deneyim kazanabileceği görevler ve başarımlar eklenebilir.

5. **Kaydetme Sistemi**: Karakter durumlarını kaydetme ve yükleme özelliği eklenebilir.
\`\`\`
`; 