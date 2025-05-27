import { Metadata } from "next";

// Ana SEO metadata yapılandırması
export const defaultMetadata: Metadata = {
  title: {
    default: "Kodleon | Yapay Zeka Eğitim Platformu",
    template: "%s | Kodleon"
  },
  description: "Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri sunan modern eğitim platformu.",
  keywords: ["yapay zeka", "makine öğrenmesi", "derin öğrenme", "yapay zeka eğitimi", "kodleon", "türkçe ai eğitimi", "veri bilimi", "nlp", "doğal dil işleme", "bilgisayarlı görü"],
  authors: [{ name: "Kodleon Ekibi" }],
  creator: "Kodleon",
  publisher: "Kodleon",
  metadataBase: new URL("https://kodleon.com"),
  openGraph: {
    type: "website",
    locale: "tr_TR",
    url: "https://kodleon.com",
    title: "Kodleon | Yapay Zeka Eğitim Platformu",
    description: "Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri sunan modern eğitim platformu.",
    siteName: "Kodleon",
  },
  twitter: {
    card: "summary_large_image",
    title: "Kodleon | Yapay Zeka Eğitim Platformu",
    description: "Yapay zeka, makine öğrenmesi ve veri bilimi konularında Türkçe eğitim içerikleri.",
    creator: "@kodleon",
  },
  alternates: {
    canonical: "https://kodleon.com",
  },
  robots: {
    index: true,
    follow: true,
  },
};

// Sayfa türüne göre metadata oluşturma
export function createPageMetadata({
  title,
  description,
  path,
  keywords = [],
  imageUrl,
}: {
  title: string;
  description: string;
  path: string;
  keywords?: string[];
  imageUrl?: string;
}): Metadata {
  const url = `https://kodleon.com${path}`;
  const ogImageUrl = imageUrl || "https://kodleon.com/og-image.jpg";

  return {
    title,
    description,
    keywords: [...defaultMetadata.keywords as string[], ...keywords],
    alternates: {
      canonical: url,
    },
    openGraph: {
      title,
      description,
      url,
      images: [
        {
          url: ogImageUrl,
          width: 1200,
          height: 630,
          alt: `${title} - Kodleon Yapay Zeka Eğitim Platformu`,
        },
      ],
    },
    twitter: {
      title,
      description,
      images: [ogImageUrl],
    },
  };
}

// Kategori/konu sayfaları için metadata oluşturma
export function createTopicPageMetadata({
  topicName,
  topicSlug,
  description,
  keywords = [],
  imageUrl,
}: {
  topicName: string;
  topicSlug: string;
  description: string;
  keywords?: string[];
  imageUrl?: string;
}): Metadata {
  return createPageMetadata({
    title: `${topicName} Eğitimi`,
    description,
    path: `/topics/${topicSlug}`,
    keywords,
    imageUrl,
  });
}

// JSON-LD Yapıları
type SchemaOrgData = {
  [key: string]: any;
};

// Eğitim kursu şeması
export function createCourseSchema(course: {
  name: string;
  description: string;
  provider?: string;
  url: string;
  imageUrl?: string;
  instructor?: string;
  keywords?: string[];
}): SchemaOrgData {
  return {
    "@context": "https://schema.org",
    "@type": "Course",
    name: course.name,
    description: course.description,
    provider: {
      "@type": "Organization",
      name: course.provider || "Kodleon",
      sameAs: "https://kodleon.com",
    },
    url: course.url,
    ...(course.imageUrl && { image: course.imageUrl }),
    ...(course.instructor && {
      instructor: {
        "@type": "Person",
        name: course.instructor,
      },
    }),
    inLanguage: "tr-TR",
    ...(course.keywords && { keywords: course.keywords.join(", ") }),
  };
}

// Makale şeması
export function createArticleSchema(article: {
  headline: string;
  description: string;
  authorName: string;
  publishDate: string;
  modifiedDate?: string;
  imageUrl?: string;
  url: string;
}): SchemaOrgData {
  return {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: article.headline,
    description: article.description,
    author: {
      "@type": "Person",
      name: article.authorName,
    },
    publisher: {
      "@type": "Organization",
      name: "Kodleon",
      logo: {
        "@type": "ImageObject",
        url: "https://kodleon.com/logo.png",
      },
    },
    datePublished: article.publishDate,
    dateModified: article.modifiedDate || article.publishDate,
    ...(article.imageUrl && { image: article.imageUrl }),
    mainEntityOfPage: {
      "@type": "WebPage",
      "@id": article.url,
    },
    inLanguage: "tr-TR",
  };
}

// Soru-Cevap şeması (SSS sayfaları için)
export function createFAQSchema(questions: {
  question: string;
  answer: string;
}[]): SchemaOrgData {
  return {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: questions.map((q) => ({
      "@type": "Question",
      name: q.question,
      acceptedAnswer: {
        "@type": "Answer",
        text: q.answer,
      },
    })),
  };
} 