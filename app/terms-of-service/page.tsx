import { Metadata } from 'next';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Kullanım Şartları | Kodleon',
  description: 'Kodleon web sitesi ve hizmetleri için kullanım şartları.',
  openGraph: {
    title: 'Kullanım Şartları | Kodleon',
    description: 'Kodleon web sitesi ve hizmetleri için kullanım şartları.',
    url: 'https://kodleon.com/terms-of-service',
    images: [
      {
        url: '/images/og-image.png', // Genel OG görseli
        width: 1200,
        height: 630,
        alt: 'Kodleon Kullanım Şartları'
      }
    ]
  }
};

export default function TermsOfServicePage() {
  return (
    <div className="bg-background text-foreground py-12 md:py-20">
      <div className="container max-w-4xl mx-auto px-4">
        <header className="mb-10 md:mb-12 text-center">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-primary">
            Kullanım Şartları
          </h1>
          <p className="text-lg text-muted-foreground mt-2">
            Son Güncelleme: 31 Mayıs 2025
          </p>
        </header>

        <article className="prose prose-lg dark:prose-invert mx-auto">
          <p>
            Lütfen Kodleon web sitesini ('Site') kullanmadan önce bu Kullanım Şartlarını ('Şartlar', 'Kullanım Şartları') dikkatlice okuyun. Siteye erişiminiz ve Siteyi kullanımınız, bu Şartları kabul etmenize ve bunlara uymanıza bağlıdır. Bu Şartlar, Siteye erişen veya Siteyi kullanan tüm ziyaretçiler, kullanıcılar ve diğerleri için geçerlidir.
          </p>

          <h2>Hesaplar</h2>
          <p>
            Bizde bir hesap oluşturduğunuzda, bize her zaman doğru, eksiksiz ve güncel bilgiler sağlamalısınız. Bunun yapılmaması, Şartların ihlali anlamına gelir ve Sitemizdeki hesabınızın derhal feshedilmesine neden olabilir.
          </p>
          <p>
            Parolanızı korumaktan ve parolanızın Sitemizde veya bir üçüncü taraf hizmetinde olmasına bakılmaksızın parolanız altındaki tüm faaliyetlerden veya eylemlerden siz sorumlusunuz.
          </p>

          <h2>Fikri Mülkiyet</h2>
          <p>
            Site ve orijinal içeriği, özellikleri ve işlevselliği Kodleon ve lisans verenlerinin münhasır mülkiyetindedir ve öyle kalacaktır. Site, hem Türkiye'nin hem de yabancı ülkelerin telif hakkı, ticari marka ve diğer yasalarıyla korunmaktadır.
          </p>

          <h2>Diğer Web Sitelerine Bağlantılar</h2>
          <p>
            Sitemiz, Kodleon'a ait olmayan veya Kodleon tarafından kontrol edilmeyen üçüncü taraf web sitelerine veya hizmetlerine bağlantılar içerebilir.
          </p>
          <p>
            Kodleon'un herhangi bir üçüncü taraf web sitesinin veya hizmetinin içeriği, gizlilik politikaları veya uygulamaları üzerinde hiçbir kontrolü yoktur ve bunlar için hiçbir sorumluluk kabul etmez. Ayrıca, bu tür web sitelerinde veya hizmetlerinde bulunan veya bunlar aracılığıyla elde edilen herhangi bir içerik, mal veya hizmetin kullanımından veya bunlara güvenilmesinden kaynaklanan veya kaynaklandığı iddia edilen herhangi bir hasar veya kayıptan doğrudan veya dolaylı olarak Kodleon'un sorumlu veya yükümlü olmayacağını kabul edersiniz.
          </p>

          <h2>Fesih</h2>
          <p>
            Bu Şartları ihlal etmeniz de dahil olmak üzere, herhangi bir nedenle, önceden bildirimde bulunmaksızın veya yükümlülük altına girmeksizin Sitemize erişimi derhal feshedebilir veya askıya alabiliriz.
          </p>

          <h2>Sorumluluğun Sınırlandırılması</h2>
          <p>
            Kodleon, yöneticileri, çalışanları, ortakları, acenteleri, tedarikçileri veya iştirakleri, hiçbir durumda, (i) Siteye erişiminizden veya Siteyi kullanmanızdan veya Siteye erişememenizden veya Siteyi kullanamamanızdan; (ii) Sitedeki herhangi bir üçüncü tarafın herhangi bir davranışı veya içeriğinden; (iii) Siteden elde edilen herhangi bir içerikten; ve (iv) yetkisiz erişim, kullanım veya iletimlerinizin veya içeriğinizin değiştirilmesinden kaynaklanan, kar kaybı, veri kaybı, kullanım kaybı, iyi niyet kaybı veya diğer soyut kayıplar dahil ancak bunlarla sınırlı olmamak üzere dolaylı, arızi, özel, sonuç olarak ortaya çıkan veya cezai zararlardan sorumlu olmayacaktır.
          </p>

          <h2>Uygulanacak Hukuk</h2>
          <p>
            Bu Şartlar, kanunlar ihtilafı hükümlerine bakılmaksızın Türkiye Cumhuriyeti kanunlarına göre yönetilecek ve yorumlanacaktır.
          </p>

          <h2>Değişiklikler</h2>
          <p>
            Tamamen kendi takdirimize bağlı olarak, bu Şartları herhangi bir zamanda değiştirme veya yerine yenilerini getirme hakkımızı saklı tutarız. Bir revizyon önemliyse, yeni şartların yürürlüğe girmesinden en az 30 gün önce bildirimde bulunmaya çalışacağız. Neyin önemli bir değişiklik teşkil edeceği tamamen kendi takdirimize bağlı olarak belirlenecektir.
          </p>

          <h2>Bize Ulaşın</h2>
          <p>
            Bu Şartlar hakkında herhangi bir sorunuz varsa, lütfen bizimle iletişime geçin:
          </p>
          <p>
            E-posta: <Link href="mailto:info@kodleon.com">info@kodleon.com</Link>
          </p>
        </article>
      </div>
    </div>
  );
} 