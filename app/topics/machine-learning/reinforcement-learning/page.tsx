import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const metadata: Metadata = createPageMetadata({
  title: 'Pekiştirmeli Öğrenme',
  description: 'Kodleon'da pekiştirmeli öğrenme prensiplerini ve ajanların ödül-ceza sistemiyle nasıl öğrendiğini keşfedin.',
  path: '/topics/machine-learning/reinforcement-learning',
  keywords: ['pekiştirmeli öğrenme', 'reinforcement learning', 'makine öğrenmesi', 'ajanlar', 'ödül', 'ceza', 'markov karar süreçleri', 'derin pekiştirmeli öğrenme', 'kodleon', 'türkçe ai eğitimi'],
});

export default function ReinforcementLearningPage() {
  return (
    <div className="container max-w-6xl mx-auto py-12">
      <div className="mb-8">
        <Button asChild variant="ghost" size="sm" className="gap-1">
          <Link href="/topics/machine-learning" aria-label="Makine Öğrenmesi konusuna geri dön">
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Makine Öğrenmesi
          </Link>
        </Button>
      </div>
      <h1 className="text-4xl font-bold mb-6">Pekiştirmeli Öğrenme</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Pekiştirmeli öğrenme (RL), bir yazılım ajanının (agent) bir ortamda (environment) belirli bir hedefi gerçekleştirmek için nasıl davranması gerektiğini, deneme-yanılma yoluyla ve aldığı geri bildirimlere (ödüller veya cezalar) dayanarak öğrendiği bir makine öğrenmesi paradigmıdır. Bu yaklaşımda, ajan eylemler gerçekleştirir, ortam bu eylemlere tepki verir ve ajana bir ödül sinyali gönderir. Ajanın amacı, zaman içinde toplam ödül miktarını maksimize edecek bir strateji (policy) öğrenmektir.
      </p>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h2>Temel Kavramlar</h2>
        <p>Pekiştirmeli öğrenmenin anlaşılması için bazı temel kavramlar önemlidir:</p>
        <ol>
          <li>**Ajan (Agent):** Öğrenen ve eylemler gerçekleştiren varlıktır.</li>
          <li>**Ortam (Environment):** Ajanın etkileşimde bulunduğu dış dünyadır.</li>
          <li>**Durum (State):** Ortamın belirli bir zamandaki anlık konfigürasyonudur.</li>
          <li>**Eylem (Action):** Ajanın belirli bir durumda alabileceği kararlardır.</li>
          <li>**Ödül (Reward):** Ajanın bir eylem sonucunda ortamdan aldığı geri bildirimdir; genellikle sayısal bir değerdir.</li>
          <li>**Politika (Policy):** Ajanın belirli bir durumda hangi eylemi seçeceğini belirleyen stratejidir.</li>
          <li>**Değer Fonksiyonu (Value Function):** Belirli bir durumdan başlayarak veya belirli bir eylemi belirli bir durumda gerçekleştirerek elde edilmesi beklenen gelecekteki toplam ödül miktarını tahmin eder.</li>
        </ol>

        <h2>Algoritmalar ve Yaklaşımlar</h2>
        <p>Pekiştirmeli öğrenmede çeşitli algoritmalar kullanılır:</p>
        <ul>
          <li>**Q-Learning:** Değer tabanlı bir algoritma olup, her durum-eylem çifti için bir Q-değeri öğrenir.</li>
          <li>**SARSA:** Q-learning'e benzer, ancak bir sonraki eylemi de dikkate alarak Q-değerlerini günceller.</li>
          <li>**Derin Q Ağları (DQN):** Q-learning'i derin sinir ağları ile birleştirerek karmaşık durum alanlarına sahip problemlerin çözümüne olanak tanır.</li>
          <li>**Politika Gradyanları (Policy Gradients):** Doğrudan politikayı optimize eden algoritmalardır.</li>
          <li>**Aktör-Kritik (Actor-Critic) Metotlar:** Hem politikayı hem de değer fonksiyonunu öğrenen melez yaklaşımlardır.</li>
        </ul>

        <h2>Uygulama Alanları</h2>
        <p>Pekiştirmeli öğrenme özellikle şu alanlarda etkilidir:</p>
        <ul>
          <li>**Oyun Oynama:** Satranç, Go ve video oyunları gibi karmaşık oyunlarda insanüstü performans sergileme (DeepMind'ın AlphaGo'su gibi).</li>
          <li>**Robotik:** Robotların karmaşık hareketleri ve görevleri öğrenmesi.</li>
          <li>**Otonom Sistemler:** Kendi kendine giden araçlar ve dronlar.</li>
          <li>**Kaynak Yönetimi:** Enerji şebekeleri veya veri merkezlerinde kaynak dağılımını optimize etme.</li>
          <li>**Finans:** Ticaret stratejileri geliştirme.</li>
        </ul>

        <p>Pekiştirmeli öğrenme, dinamik ortamlarda karar verme ve strateji geliştirme yeteneği sayesinde gelecekte birçok alanda çığır açma potansiyeli taşımaktadır.</p>
      </div>
    </div>
  );
} 