import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { ThemeProvider } from '@/components/theme-provider';
import Navbar from '@/components/navbar';
import Footer from '@/components/footer';
import { siteConfig } from '@/config/site';
import 'highlight.js/styles/github.css';
import Script from 'next/script';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  metadataBase: new URL('https://kodleon.com'),
  title: {
    default: siteConfig.name,
    template: `%s - ${siteConfig.name}`,
  },
  description: siteConfig.description,
  keywords: [
    'Python',
    'programlama',
    'nesne tabanlı programlama',
    'OOP',
    'yazılım geliştirme',
    'eğitim',
    'Türkçe programlama',
    'Python dersleri',
    'yazılım dersleri',
    'kodlama öğren'
  ],
  authors: [{ name: 'Kodleon Team' }],
  creator: 'Kodleon',
  publisher: 'Kodleon',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    title: 'Kodleon - Python ve Programlama Eğitimi',
    description: 'Python programlama, nesne tabanlı programlama, veri yapıları ve algoritmalar hakkında kapsamlı Türkçe eğitim içerikleri.',
    url: 'https://kodleon.com',
    siteName: 'Kodleon',
    locale: 'tr_TR',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Kodleon - Python ve Programlama Eğitimi',
    description: 'Python programlama, nesne tabanlı programlama, veri yapıları ve algoritmalar hakkında kapsamlı Türkçe eğitim içerikleri.',
    creator: '@kodleon',
  },
  verification: {
    google: 'google-site-verification-code',
  },
  alternates: {
    canonical: 'https://kodleon.com',
  },
  robots: {
    index: true,
    follow: true,
  },
  icons: {
    icon: [
      { url: '/images/favicons/favicon.svg', type: 'image/svg+xml' },
      { url: '/images/favicons/favicon-16x16.png', sizes: '16x16', type: 'image/png' }
    ],
    apple: [
      { url: '/images/favicons/favicon.svg', type: 'image/svg+xml' }
    ],
    shortcut: '/images/favicons/favicon.svg'
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
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="tr" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/images/favicons/favicon.svg" type="image/svg+xml" />
        <link rel="icon" href="/images/favicons/favicon-16x16.png" sizes="16x16" type="image/png" />
        <link
          rel="apple-touch-icon"
          href="/images/favicons/favicon.svg"
          type="image/svg+xml"
        />
        <meta name="google-site-verification" content="your-verification-code" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "EducationalOrganization",
              "name": "Kodleon",
              "url": "https://kodleon.com",
              "logo": "https://kodleon.com/images/logo.jpg",
              "description": "Python programlama, nesne tabanlı programlama, veri yapıları ve algoritmalar hakkında kapsamlı Türkçe eğitim içerikleri.",
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
        
        {/* Client-side script to handle www to non-www redirection (replacement for middleware) */}
        <Script id="www-redirect" strategy="beforeInteractive">
          {`
            (function() {
              // Only run in browser, not during static generation
              if (typeof window !== 'undefined') {
                var host = window.location.host;
                if (host.startsWith('www.')) {
                  var newHost = host.replace(/^www\\./, '');
                  window.location.href = window.location.protocol + '//' + newHost + window.location.pathname + window.location.search;
                }
              }
            })();
          `}
        </Script>
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