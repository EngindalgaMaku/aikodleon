export const content = `
# Medya İşleme Sistemi Alıştırması

Bu alıştırmada, farklı medya türlerini (müzik, video, podcast) işleyebilen bir sistem tasarlayacağız. Sistem, soyut sınıflar ve arayüzler kullanarak medya işleme işlevselliğini sağlayacak.

## Problem Tanımı

Aşağıdaki özelliklere sahip bir medya işleme sistemi geliştirmeniz gerekiyor:

1. Farklı medya türleri için ortak bir arayüz
2. Her medya türü için özel oynatma davranışları
3. Format dönüştürme işlemleri
4. Kalite ayarları
5. Çalma listesi yönetimi

## Çözüm

\`\`\`python
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

# Medya türleri için enum
class MediaType(Enum):
    MUSIC = "music"
    VIDEO = "video"
    PODCAST = "podcast"

# Medya formatları için enum
class MediaFormat(Enum):
    MP3 = "mp3"
    WAV = "wav"
    MP4 = "mp4"
    MKV = "mkv"
    M4A = "m4a"

# Kalite ayarları için enum
class Quality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Temel medya oynatıcı sınıfı
class MediaPlayer(ABC):
    def __init__(self, media_type: MediaType):
        self.media_type = media_type
        self.is_playing = False
        self.current_position = 0
        self.volume = 50
        
    @abstractmethod
    def play(self) -> bool:
        pass
    
    @abstractmethod
    def pause(self) -> bool:
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        pass
    
    def set_volume(self, volume: int) -> bool:
        if 0 <= volume <= 100:
            self.volume = volume
            return True
        return False
    
    def seek(self, position: int) -> bool:
        if position >= 0:
            self.current_position = position
            return True
        return False
    
    @abstractmethod
    def get_info(self) -> dict:
        pass

# Müzik oynatıcı
class MusicPlayer(MediaPlayer):
    def __init__(self):
        super().__init__(MediaType.MUSIC)
        self.equalizer_enabled = False
        self.equalizer_settings = {
            "bass": 0,
            "treble": 0,
            "mid": 0
        }
    
    def play(self) -> bool:
        self.is_playing = True
        print(f"Playing music at position {self.current_position}")
        return True
    
    def pause(self) -> bool:
        self.is_playing = False
        print("Music paused")
        return True
    
    def stop(self) -> bool:
        self.is_playing = False
        self.current_position = 0
        print("Music stopped")
        return True
    
    def set_equalizer(self, bass: int, treble: int, mid: int) -> bool:
        self.equalizer_settings = {
            "bass": bass,
            "treble": treble,
            "mid": mid
        }
        return True
    
    def get_info(self) -> dict:
        return {
            "type": self.media_type.value,
            "playing": self.is_playing,
            "position": self.current_position,
            "volume": self.volume,
            "equalizer": self.equalizer_settings
        }

# Video oynatıcı
class VideoPlayer(MediaPlayer):
    def __init__(self):
        super().__init__(MediaType.VIDEO)
        self.resolution = "1080p"
        self.subtitles_enabled = False
        self.subtitle_language = None
    
    def play(self) -> bool:
        self.is_playing = True
        print(f"Playing video at position {self.current_position}")
        return True
    
    def pause(self) -> bool:
        self.is_playing = False
        print("Video paused")
        return True
    
    def stop(self) -> bool:
        self.is_playing = False
        self.current_position = 0
        print("Video stopped")
        return True
    
    def set_resolution(self, resolution: str) -> bool:
        self.resolution = resolution
        return True
    
    def toggle_subtitles(self, language: Optional[str] = None) -> bool:
        self.subtitles_enabled = not self.subtitles_enabled
        if language:
            self.subtitle_language = language
        return True
    
    def get_info(self) -> dict:
        return {
            "type": self.media_type.value,
            "playing": self.is_playing,
            "position": self.current_position,
            "volume": self.volume,
            "resolution": self.resolution,
            "subtitles": {
                "enabled": self.subtitles_enabled,
                "language": self.subtitle_language
            }
        }

# Podcast oynatıcı
class PodcastPlayer(MediaPlayer):
    def __init__(self):
        super().__init__(MediaType.PODCAST)
        self.playback_speed = 1.0
        self.chapter_markers = []
        self.current_chapter = 0
    
    def play(self) -> bool:
        self.is_playing = True
        print(f"Playing podcast at position {self.current_position}")
        return True
    
    def pause(self) -> bool:
        self.is_playing = False
        print("Podcast paused")
        return True
    
    def stop(self) -> bool:
        self.is_playing = False
        self.current_position = 0
        print("Podcast stopped")
        return True
    
    def set_playback_speed(self, speed: float) -> bool:
        if 0.5 <= speed <= 3.0:
            self.playback_speed = speed
            return True
        return False
    
    def add_chapter_marker(self, position: int, title: str) -> bool:
        self.chapter_markers.append({"position": position, "title": title})
        return True
    
    def get_info(self) -> dict:
        return {
            "type": self.media_type.value,
            "playing": self.is_playing,
            "position": self.current_position,
            "volume": self.volume,
            "playback_speed": self.playback_speed,
            "chapters": self.chapter_markers,
            "current_chapter": self.current_chapter
        }

# Çalma listesi yönetimi
class Playlist:
    def __init__(self, name: str):
        self.name = name
        self.items: List[MediaPlayer] = []
        self.current_index = 0
    
    def add_item(self, item: MediaPlayer) -> bool:
        self.items.append(item)
        return True
    
    def remove_item(self, index: int) -> bool:
        if 0 <= index < len(self.items):
            self.items.pop(index)
            return True
        return False
    
    def play_current(self) -> bool:
        if self.items:
            return self.items[self.current_index].play()
        return False
    
    def next(self) -> bool:
        if self.current_index < len(self.items) - 1:
            self.current_index += 1
            return self.play_current()
        return False
    
    def previous(self) -> bool:
        if self.current_index > 0:
            self.current_index -= 1
            return self.play_current()
        return False
    
    def show_playlist(self) -> List[dict]:
        return [item.get_info() for item in self.items]

# Format dönüştürücü
class MediaConverter:
    @staticmethod
    def convert(media: MediaPlayer, target_format: MediaFormat) -> bool:
        source_type = media.media_type
        print(f"Converting {source_type.value} to {target_format.value}")
        
        # Format uyumluluğunu kontrol et
        valid_formats = {
            MediaType.MUSIC: [MediaFormat.MP3, MediaFormat.WAV],
            MediaType.VIDEO: [MediaFormat.MP4, MediaFormat.MKV],
            MediaType.PODCAST: [MediaFormat.MP3, MediaFormat.M4A]
        }
        
        if target_format in valid_formats[source_type]:
            print(f"Conversion completed: {target_format.value}")
            return True
        else:
            print(f"Invalid format {target_format.value} for {source_type.value}")
            return False

# Kullanım örneği
def main():
    # Medya oynatıcıları oluştur
    music = MusicPlayer()
    video = VideoPlayer()
    podcast = PodcastPlayer()
    
    # Müzik ayarları
    music.set_equalizer(bass=5, treble=3, mid=0)
    music.set_volume(70)
    
    # Video ayarları
    video.set_resolution("4K")
    video.toggle_subtitles("TR")
    
    # Podcast ayarları
    podcast.set_playback_speed(1.5)
    podcast.add_chapter_marker(0, "Giriş")
    podcast.add_chapter_marker(300, "Ana Konu")
    podcast.add_chapter_marker(600, "Sonuç")
    
    # Çalma listesi oluştur
    playlist = Playlist("Karışık Liste")
    playlist.add_item(music)
    playlist.add_item(video)
    playlist.add_item(podcast)
    
    # Çalma listesini oynat
    print("\\nÇalma Listesi İçeriği:")
    for item in playlist.show_playlist():
        print(f"- {item['type']}")
    
    print("\\nÇalma listesi oynatılıyor...")
    playlist.play_current()  # Müzik
    playlist.next()         # Video
    playlist.next()         # Podcast
    
    # Format dönüştürme
    print("\\nFormat dönüştürme testleri:")
    converter = MediaConverter()
    converter.convert(music, MediaFormat.WAV)      # Geçerli
    converter.convert(video, MediaFormat.MP3)      # Geçersiz
    converter.convert(podcast, MediaFormat.M4A)    # Geçerli

if __name__ == "__main__":
    main()
\`\`\`

## Önemli Noktlar

1. **Soyut Temel Sınıf**: \`MediaPlayer\` sınıfı, tüm medya oynatıcılar için ortak davranışları tanımlar.
2. **Özel Özellikler**: Her medya türü için özel özellikler eklenmiştir:
   - Müzik: Ekolayzer ayarları
   - Video: Çözünürlük ve altyazı desteği
   - Podcast: Oynatma hızı ve bölüm işaretleri
3. **Tip Güvenliği**: Enum sınıfları kullanılarak tip güvenliği sağlanmıştır.
4. **Çalma Listesi**: Farklı medya türlerini tek bir listede yönetebilme.
5. **Format Dönüştürme**: Her medya türü için uygun format dönüşümlerinin kontrolü.

## Geliştirme Önerileri

1. **Ağ Desteği**: Medya dosyalarının uzaktan yüklenmesi ve akışı için destek.
2. **Önbellek Mekanizması**: Sık kullanılan medya dosyaları için önbellek sistemi.
3. **Filtreler ve Efektler**: Her medya türü için özel efekt ve filtreler.
4. **Metadata Desteği**: ID3 etiketleri, video meta bilgileri gibi metadata işleme.
5. **Çoklu Dil Desteği**: Arayüz ve içerik için çoklu dil desteği.

Bu örnek, soyut sınıflar ve arayüzlerin gerçek dünya uygulamalarında nasıl kullanılabileceğini göstermektedir. Sistem, yeni medya türleri veya özellikler eklemek için kolayca genişletilebilir şekilde tasarlanmıştır.
`; 