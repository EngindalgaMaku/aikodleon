import { Metadata } from 'next';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Gizlilik Politikası | Kodleon',
  description: 'Kodleon web sitesi ve hizmetleri için gizlilik politikası.',
  openGraph: {
    title: 'Gizlilik Politikası | Kodleon',
    description: 'Kodleon web sitesi ve hizmetleri için gizlilik politikası.',
    url: 'https://kodleon.com/privacy-policy',
    images: [
      {
        url: '/images/og-image.png', // Genel OG görseli
        width: 1200,
        height: 630,
        alt: 'Kodleon Gizlilik Politikası'
      }
    ]
  }
};

export default function PrivacyPolicyPage() {
  return (
    <div className="bg-background text-foreground py-12 md:py-20">
      <div className="container max-w-4xl mx-auto px-4">
        <header className="mb-10 md:mb-12 text-center">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-primary">
            Gizlilik Politikası
          </h1>
          <p className="text-lg text-muted-foreground mt-2">
            Son Güncelleme: 31 Mayıs 2025
          </p>
        </header>

        <article className="prose prose-lg dark:prose-invert mx-auto">
          <p>
            Kodleon ('biz', 'bizim' veya 'sitemiz') olarak gizliliğinize değer veriyoruz. Bu gizlilik politikası, web sitemizi ziyaret ettiğinizde veya hizmetlerimizi kullandığınızda kişisel bilgilerinizi nasıl topladığımızı, kullandığımızı, ifşa ettiğimizi ve koruduğumuzu açıklamaktadır.
          </p>

          <h2>Topladığımız Bilgiler</h2>
          <p>
            Sitemizi kullandığınızda sizden çeşitli türlerde bilgi toplayabiliriz:
          </p>
          <ul>
            <li><strong>Kişisel Kimlik Bilgileri:</strong> Adınız, e-posta adresiniz, telefon numaranız gibi bize doğrudan sağladığınız bilgiler.</li>
            <li><strong>Kullanım Verileri:</strong> IP adresiniz, tarayıcı türünüz, ziyaret ettiğiniz sayfalar, ziyaret süreniz gibi sitemizle nasıl etkileşim kurduğunuza dair bilgiler.</li>
            <li><strong>Çerezler ve Takip Teknolojileri:</strong> Deneyiminizi geliştirmek için çerezler ve benzeri takip teknolojileri kullanıyoruz.</li>
          </ul>

          <h2>Bilgilerinizi Nasıl Kullanıyoruz?</h2>
          <p>
            Topladığımız bilgileri aşağıdaki amaçlar için kullanabiliriz:
          </p>
          <ul>
            <li>Hizmetlerimizi sağlamak ve sürdürmek.</li>
            <li>Sitemizi kişiselleştirmek ve geliştirmek.</li>
            <li>Sizinle iletişim kurmak, sorularınıza yanıt vermek.</li>
            <li>Pazarlama ve tanıtım faaliyetleri yürütmek (onayınızla).</li>
            <li>Yasal yükümlülüklerimizi yerine getirmek.</li>
          </ul>

          <h2>Bilgilerinizi Kimlerle Paylaşıyoruz?</h2>
          <p>
            Kişisel bilgilerinizi aşağıdaki durumlar dışında üçüncü taraflarla paylaşmayız:
          </p>
          <ul>
            <li>Onayınızla.</li>
            <li>Hizmet sağlayıcılarımızla (sadece hizmetlerini yerine getirebilmeleri için gerekli ölçüde).</li>
            <li>Yasal gereklilikler veya yasal süreçlere yanıt olarak.</li>
            <li>Haklarımızı, gizliliğimizi, güvenliğimizi veya mülkiyetimizi korumak için.</li>
          </ul>

          <h2>Veri Güvenliği</h2>
          <p>
            Kişisel bilgilerinizin güvenliğini korumak için makul idari, teknik ve fiziksel güvenlik önlemleri alıyoruz. Ancak, internet üzerinden hiçbir iletim yönteminin veya elektronik depolama yönteminin %100 güvenli olmadığını lütfen unutmayın.
          </p>

          <h2>Çocukların Gizliliği</h2>
          <p>
            Hizmetlerimiz 13 yaşın altındaki çocuklara yönelik değildir. Bilerek 13 yaşın altındaki çocuklardan kişisel bilgi toplamıyoruz.
          </p>

          <h2>Gizlilik Politikasındaki Değişiklikler</h2>
          <p>
            Bu gizlilik politikasını zaman zaman güncelleyebiliriz. Herhangi bir değişiklik yaptığımızda, bu sayfada yeni gizlilik politikasını yayınlayarak sizi bilgilendireceğiz. Değişiklikler için bu gizlilik politikasını periyodik olarak gözden geçirmeniz önerilir.
          </p>

          <h2>Bize Ulaşın</h2>
          <p>
            Bu Gizlilik Politikası hakkında herhangi bir sorunuz varsa, lütfen bizimle iletişime geçin:
          </p>
          <p>
            E-posta: <Link href="mailto:info@kodleon.com">info@kodleon.com</Link>
          </p>
        </article>
      </div>
    </div>
  );
} 