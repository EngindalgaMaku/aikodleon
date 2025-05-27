import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { ThemeProvider } from '@/components/theme-provider';
import Navbar from '@/components/navbar';
import Footer from '@/components/footer';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  metadataBase: new URL('https://kodleon.com'),
  title: {
    default: 'Kodleon | Yapay Zeka Eğitim Platformu',
    template: '%s | Kodleon'
  },
  description: 'Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri sunan modern eğitim platformu.',
  keywords: ['yapay zeka', 'makine öğrenmesi', 'derin öğrenme', 'yapay zeka eğitimi', 'kodleon', 'türkçe ai eğitimi', 'veri bilimi', 'nlp', 'doğal dil işleme', 'bilgisayarlı görü'],
  authors: [{ name: 'Kodleon Ekibi' }],
  creator: 'Kodleon',
  publisher: 'Kodleon',
  openGraph: {
    type: 'website',
    locale: 'tr_TR',
    url: 'https://kodleon.com',
    title: 'Kodleon | Yapay Zeka Eğitim Platformu',
    description: 'Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri sunan modern eğitim platformu.',
    siteName: 'Kodleon',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Kodleon | Yapay Zeka Eğitim Platformu',
    description: 'Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri.',
    creator: '@kodleon',
  },
  alternates: {
    canonical: 'https://kodleon.com',
  },
  robots: {
    index: true,
    follow: true,
  },
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-icon.png',
  },
  verification: {
    google: 'google-site-verification-code',
  },
  category: 'education',
  formatDetection: {
    telephone: false,
  },
  viewport: {
    width: 'device-width',
    initialScale: 1,
  },
  applicationName: 'Kodleon',
  colorScheme: 'dark light',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#000000' },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="tr" suppressHydrationWarning>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "EducationalOrganization",
              "name": "Kodleon",
              "url": "https://kodleon.com",
              "logo": "https://kodleon.com/logo.png",
              "description": "Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri sunan modern eğitim platformu.",
              "sameAs": [
                "https://twitter.com/kodleon",
                "https://www.linkedin.com/company/kodleon",
                "https://www.youtube.com/kodleon"
              ],
              "address": {
                "@type": "PostalAddress",
                "addressCountry": "TR"
              },
              "contactPoint": {
                "@type": "ContactPoint",
                "contactType": "customer service",
                "email": "info@kodleon.com"
              }
            })
          }}
        />
      </head>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="relative flex min-h-screen flex-col">
            <header>
              <Navbar />
            </header>
            <main className="flex-1">{children}</main>
            <Footer />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}