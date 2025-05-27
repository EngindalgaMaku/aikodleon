import Script from "next/script";

export default function Head() {
  return (
    <>
      <link rel="alternate" href="https://kodleon.com" hrefLang="tr-TR" />
      <meta name="geo.region" content="TR" />
      <meta name="geo.placename" content="Türkiye" />
      <meta name="content-language" content="tr" />
      <meta name="google" content="notranslate" />
      
      <Script id="schema-kodleon" type="application/ld+json">
        {`
          {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "url": "https://kodleon.com/",
            "name": "Kodleon - Yapay Zeka Eğitim Platformu",
            "description": "Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri sunan modern eğitim platformu.",
            "potentialAction": {
              "@type": "SearchAction",
              "target": {
                "@type": "EntryPoint",
                "urlTemplate": "https://kodleon.com/search?q={search_term_string}"
              },
              "query-input": "required name=search_term_string"
            },
            "inLanguage": "tr-TR"
          }
        `}
      </Script>
      
      <Script id="schema-organization" type="application/ld+json">
        {`
          {
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": "Kodleon",
            "url": "https://kodleon.com",
            "logo": "https://kodleon.com/logo.png",
            "sameAs": [
              "https://twitter.com/kodleon",
              "https://www.linkedin.com/company/kodleon",
              "https://www.youtube.com/kodleon"
            ],
            "contactPoint": {
              "@type": "ContactPoint",
              "telephone": "",
              "contactType": "customer service",
              "email": "info@kodleon.com",
              "areaServed": "TR",
              "availableLanguage": "Turkish"
            }
          }
        `}
      </Script>
      
      <Script id="google-tag-manager" strategy="afterInteractive">
        {`
          (function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
          new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
          j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
          'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
          })(window,document,'script','dataLayer','GTM-XXXXXXX');
        `}
      </Script>
    </>
  );
} 