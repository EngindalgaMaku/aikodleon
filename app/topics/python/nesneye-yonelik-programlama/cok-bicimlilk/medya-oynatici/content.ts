export const content = `
# Medya Oynatıcı Sistemi

Bu alıştırmada, çok biçimliliği kullanarak farklı medya tiplerini (müzik, video, podcast) oynatabilecek bir sistem tasarlayacağız.

## Problem Tanımı

Farklı medya tiplerini destekleyen bir medya oynatıcı sistemi geliştirmemiz gerekiyor. Sistem şu özelliklere sahip olmalı:

* Farklı medya tipleri için ortak bir arayüz
* Her medya tipi için özel oynatma davranışları
* Format dönüşümleri ve kalite ayarları
* Çalma listesi yönetimi

## Çözüm

### 1. Temel Medya Oynatıcı Sınıfı

Önce tüm medya oynatıcıları için temel bir sınıf oluşturalım:

\`\`\`python
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

class MediaQuality(Enum):
    LOW = "düşük"
    MEDIUM = "orta"
    HIGH = "yüksek"

@dataclass
class MediaInfo:
    title: str
    artist: str
    duration: int  # saniye cinsinden
    format: str
    quality: MediaQuality

class MediaPlayer(ABC):
    def __init__(self, media_info: MediaInfo):
        self.media_info = media_info
        self.current_position = 0
        self.is_playing = False
        self.volume = 50  # 0-100 arası
    
    @abstractmethod
    def play(self) -> str:
        pass
    
    @abstractmethod
    def pause(self) -> str:
        pass
    
    @abstractmethod
    def stop(self) -> str:
        pass
    
    def set_volume(self, volume: int) -> str:
        if 0 <= volume <= 100:
            self.volume = volume
            return f"Ses seviyesi {volume} olarak ayarlandı"
        raise ValueError("Ses seviyesi 0-100 arasında olmalıdır")
    
    def seek(self, position: int) -> str:
        if 0 <= position <= self.media_info.duration:
            self.current_position = position
            return f"Pozisyon {position} saniyeye ayarlandı"
        raise ValueError("Geçersiz pozisyon")
    
    def get_info(self) -> str:
        return f"""
Medya Bilgisi:
Başlık: {self.media_info.title}
Sanatçı: {self.media_info.artist}
Süre: {self.media_info.duration} saniye
Format: {self.media_info.format}
Kalite: {self.media_info.quality.value}
"""
\`\`\`

### 2. Özel Medya Oynatıcılar

Şimdi farklı medya tipleri için özel oynatıcılar oluşturalım:

\`\`\`python
class MusicPlayer(MediaPlayer):
    def __init__(self, media_info: MediaInfo):
        super().__init__(media_info)
        self.equalizer_enabled = False
    
    def play(self) -> str:
        self.is_playing = True
        return f"🎵 Müzik çalınıyor: {self.media_info.title}"
    
    def pause(self) -> str:
        self.is_playing = False
        return "Müzik duraklatıldı"
    
    def stop(self) -> str:
        self.is_playing = False
        self.current_position = 0
        return "Müzik durduruldu"
    
    def toggle_equalizer(self) -> str:
        self.equalizer_enabled = not self.equalizer_enabled
        return f"Ekolayzer {'açık' if self.equalizer_enabled else 'kapalı'}"

class VideoPlayer(MediaPlayer):
    def __init__(self, media_info: MediaInfo):
        super().__init__(media_info)
        self.subtitle_enabled = False
        self.resolution = "1080p"
    
    def play(self) -> str:
        self.is_playing = True
        return f"▶️ Video oynatılıyor: {self.media_info.title}"
    
    def pause(self) -> str:
        self.is_playing = False
        return "Video duraklatıldı"
    
    def stop(self) -> str:
        self.is_playing = False
        self.current_position = 0
        return "Video durduruldu"
    
    def toggle_subtitles(self) -> str:
        self.subtitle_enabled = not self.subtitle_enabled
        return f"Altyazılar {'açık' if self.subtitle_enabled else 'kapalı'}"
    
    def change_resolution(self, resolution: str) -> str:
        allowed_resolutions = ["720p", "1080p", "4K"]
        if resolution in allowed_resolutions:
            self.resolution = resolution
            return f"Çözünürlük {resolution} olarak değiştirildi"
        raise ValueError("Desteklenmeyen çözünürlük")

class PodcastPlayer(MediaPlayer):
    def __init__(self, media_info: MediaInfo):
        super().__init__(media_info)
        self.playback_speed = 1.0
    
    def play(self) -> str:
        self.is_playing = True
        return f"🎙️ Podcast oynatılıyor: {self.media_info.title}"
    
    def pause(self) -> str:
        self.is_playing = False
        return "Podcast duraklatıldı"
    
    def stop(self) -> str:
        self.is_playing = False
        self.current_position = 0
        return "Podcast durduruldu"
    
    def set_playback_speed(self, speed: float) -> str:
        allowed_speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        if speed in allowed_speeds:
            self.playback_speed = speed
            return f"Oynatma hızı {speed}x olarak ayarlandı"
        raise ValueError("Desteklenmeyen oynatma hızı")
\`\`\`

### 3. Çalma Listesi Yönetimi

Farklı medya tiplerini bir arada yönetebilen bir çalma listesi sınıfı:

\`\`\`python
class Playlist:
    def __init__(self, name: str):
        self.name = name
        self.items: List[MediaPlayer] = []
        self.current_index = 0
    
    def add_item(self, item: MediaPlayer) -> str:
        self.items.append(item)
        return f"{item.media_info.title} çalma listesine eklendi"
    
    def remove_item(self, index: int) -> str:
        if 0 <= index < len(self.items):
            item = self.items.pop(index)
            return f"{item.media_info.title} çalma listesinden çıkarıldı"
        raise IndexError("Geçersiz indeks")
    
    def play_next(self) -> str:
        if not self.items:
            return "Çalma listesi boş"
        
        if self.current_index < len(self.items) - 1:
            self.current_index += 1
            return self.items[self.current_index].play()
        return "Çalma listesinin sonuna gelindi"
    
    def play_previous(self) -> str:
        if not self.items:
            return "Çalma listesi boş"
        
        if self.current_index > 0:
            self.current_index -= 1
            return self.items[self.current_index].play()
        return "Çalma listesinin başındasınız"
    
    def get_current_item(self) -> Optional[MediaPlayer]:
        if self.items:
            return self.items[self.current_index]
        return None
    
    def show_playlist(self) -> str:
        if not self.items:
            return "Çalma listesi boş"
        
        result = f"\\n{self.name} Çalma Listesi:\\n"
        for i, item in enumerate(self.items):
            current = "► " if i == self.current_index else "  "
            result += f"{current}{i+1}. {item.media_info.title} ({item.media_info.duration}s)\\n"
        return result
\`\`\`

### 4. Format Dönüştürücü

Medya dosyalarının formatını dönüştürmek için bir yardımcı sınıf:

\`\`\`python
class MediaConverter:
    @staticmethod
    def convert_format(media_player: MediaPlayer, target_format: str) -> MediaPlayer:
        supported_formats = {
            "music": [".mp3", ".wav", ".flac"],
            "video": [".mp4", ".avi", ".mkv"],
            "podcast": [".mp3", ".m4a", ".wav"]
        }
        
        # Medya tipine göre desteklenen formatları kontrol et
        if isinstance(media_player, MusicPlayer):
            formats = supported_formats["music"]
        elif isinstance(media_player, VideoPlayer):
            formats = supported_formats["video"]
        elif isinstance(media_player, PodcastPlayer):
            formats = supported_formats["podcast"]
        else:
            raise ValueError("Desteklenmeyen medya tipi")
        
        # Format kontrolü
        if not target_format.startswith("."):
            target_format = "." + target_format
        
        if target_format not in formats:
            raise ValueError(f"Desteklenmeyen format. Desteklenen formatlar: {', '.join(formats)}")
        
        # Yeni medya bilgisi oluştur
        new_info = MediaInfo(
            title=media_player.media_info.title,
            artist=media_player.media_info.artist,
            duration=media_player.media_info.duration,
            format=target_format,
            quality=media_player.media_info.quality
        )
        
        # Aynı tip medya oynatıcı ile yeni nesne oluştur
        if isinstance(media_player, MusicPlayer):
            return MusicPlayer(new_info)
        elif isinstance(media_player, VideoPlayer):
            return VideoPlayer(new_info)
        else:
            return PodcastPlayer(new_info)

### 5. Kullanım Örneği

\`\`\`python
# Medya bilgileri oluştur
song_info = MediaInfo(
    title="Bohemian Rhapsody",
    artist="Queen",
    duration=354,
    format=".mp3",
    quality=MediaQuality.HIGH
)

video_info = MediaInfo(
    title="Python Eğitimi",
    artist="Kodleon",
    duration=1200,
    format=".mp4",
    quality=MediaQuality.HIGH
)

podcast_info = MediaInfo(
    title="Teknoloji Sohbetleri",
    artist="Tech Pod",
    duration=1800,
    format=".mp3",
    quality=MediaQuality.MEDIUM
)

# Oynatıcıları oluştur
music = MusicPlayer(song_info)
video = VideoPlayer(video_info)
podcast = PodcastPlayer(podcast_info)

# Çalma listesi oluştur
playlist = Playlist("Karışık Liste")
playlist.add_item(music)
playlist.add_item(video)
playlist.add_item(podcast)

# Oynatma ve kontrol
print(playlist.show_playlist())
print(playlist.get_current_item().play())
print(playlist.play_next())

# Format dönüştürme
converter = MediaConverter()
new_music = converter.convert_format(music, "flac")
print(new_music.get_info())

# Özel fonksiyonları kullan
if isinstance(playlist.get_current_item(), VideoPlayer):
    video = playlist.get_current_item()
    print(video.toggle_subtitles())
    print(video.change_resolution("4K"))
\`\`\`

## Önemli Noktalar

1. **Soyut Temel Sınıf**: \`MediaPlayer\` sınıfı soyut bir temel sınıf olarak tasarlandı ve ortak davranışları tanımladı.

2. **Çok Biçimlilik**: Her medya tipi kendi \`play()\`, \`pause()\` ve \`stop()\` metodlarını özelleştirdi.

3. **Tip Güvenliği**: \`MediaInfo\` veri sınıfı ve \`MediaQuality\` enum sınıfı ile tip güvenliği sağlandı.

4. **Özel Özellikler**: Her medya tipi kendine özgü özelliklere sahip:
   - MusicPlayer: Ekolayzer
   - VideoPlayer: Altyazı ve çözünürlük
   - PodcastPlayer: Oynatma hızı

5. **Esnek Tasarım**: Sistem yeni medya tipleri eklenecek şekilde genişletilebilir.

## Geliştirme Önerileri

1. **Ağ Desteği**: Medya dosyalarının uzaktan yüklenmesi ve akışı için destek eklenebilir.

2. **Önbellek Sistemi**: Sık kullanılan medya dosyaları için önbellek mekanizması eklenebilir.

3. **Filtre ve Efektler**: Her medya tipi için özel filtre ve efektler eklenebilir.

4. **Metadata Desteği**: ID3 etiketleri gibi metadata okuma ve yazma desteği eklenebilir.

5. **Çoklu Dil Desteği**: Arayüz ve altyazılar için çoklu dil desteği eklenebilir.
\`\`\`
`; 