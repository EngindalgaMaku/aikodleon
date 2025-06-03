import Link from "next/link";
import Image from "next/image";
import { ArrowLeft, ArrowRight, Brain, CheckCircle2, Ship as Chip, Database, Eye, FileText, Lightbulb, Rocket, Shapes, Users, FlaskConical, Sigma, Code2, BarChart3, DatabaseZap, BrainCircuit, GitMerge, ClipboardCheck, GraduationCap, Shield, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import TopicSubtopicsList from "@/components/TopicSubtopicsList";
import PaginatedSubtopicsList from "@/components/PaginatedSubtopicsList";

// Topic data mapping
const topicsData: Record<string, any> = {
  "machine-learning": {
    title: "Makine Öğrenmesi",
    description: "Algoritmaların veri kullanarak nasıl öğrendiğini ve tahminlerde bulunduğunu keşfedin.",
    icon: <Database className="h-8 w-8 text-chart-1" />,
    imageUrl: "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Makine öğrenmesi (ML), bilgisayar sistemlerinin açıkça programlanmadan verilerden öğrenmesini ve bu öğrenme yoluyla belirli görevleri yerine getirmesini sağlayan yapay zekanın bir dalıdır. ML algoritmaları, büyük veri kümelerindeki desenleri ve ilişkileri tanımlayarak çalışır, böylece bilinmeyen veriler hakkında tahminlerde bulunabilir veya kararlar alabilirler. Denetimli öğrenme (etiketli verilerle eğitim), denetimsiz öğrenme (etiketlenmemiş verilerden desen keşfi) ve pekiştirmeli öğrenme (deneme-yanılma yoluyla öğrenme) gibi çeşitli paradigmaları içerir. Makine öğrenmesi, sadece bir dizi algoritmadan ibaret değildir; aynı zamanda problem tanımlama, veri toplama ve ön işleme, model seçimi, eğitim, değerlendirme ve dağıtım gibi iteratif bir süreçtir. Tavsiye sistemlerinden otonom araçlara, tıbbi teşhisten finansal analizlere ve doğal dil işlemeden bilgisayarlı görüye kadar çok geniş bir uygulama alanına sahiptir. Bu alanda uzmanlaşmak, günümüzün ve geleceğin teknoloji dünyasında çığır açan çözümler geliştirme ve önemli bir yer edinme anlamına gelir.",
    subtopics: [
      {
        title: "Denetimli Öğrenme",
        description: "Etiketli verilerle modellerin nasıl eğitildiğini ve tahminlerde bulunduğunu öğrenin.",
        imageUrl: "https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/machine-learning/supervised-learning"
      },
      {
        title: "Denetimsiz Öğrenme",
        description: "Etiketlenmemiş verilerden kalıpları ve yapıları nasıl keşfedeceğinizi anlayın.",
        imageUrl: "https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/machine-learning/unsupervised-learning"
      },
      {
        title: "Pekiştirmeli Öğrenme",
        description: "Deneme yanılma yoluyla ajanların çevreleriyle nasıl etkileşime girdiğini ve öğrendiğini keşfedin.",
        imageUrl: "https://images.pexels.com/photos/6153354/pexels-photo-6153354.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/machine-learning/reinforcement-learning"
      },
      {
        title: "Derin Öğrenme Temelleri",
        description: "Derin öğrenmenin temel kavramlarını ve yapay sinir ağlarının çalışma prensiplerini öğrenin.",
        imageUrl: "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/machine-learning/deep-learning-basics"
      }
    ],
    skills: ["Veri Analizi", "Python", "Algoritma Tasarımı", "Model Değerlendirme", "Veri Ön İşleme", "Denetimli Öğrenme Modelleri", "Denetimsiz Öğrenme Teknikleri", "Model Optimizasyonu", "Makine Öğrenmesi Kütüphaneleri (Scikit-learn, TensorFlow, PyTorch)"],
    resources: [
      { title: "Makine Öğrenmesi Temelleri", type: "Kurs", link: "#" },
      { title: "Scikit-Learn ile Uygulamalı ML", type: "Pratik", link: "#" },
      { title: "Makine Öğrenmesi Algoritmaları Derinlemesine İnceleme", type: "E-Kitap", link: "#" },
      { title: "Kaggle ML Yarışmalarına Giriş", type: "Kaynak", link: "#" }
    ]
  },
  "nlp": {
    title: "Doğal Dil İşleme",
    description: "Makinelerin insan dilini nasıl anlayıp işlediğini ve ürettiğini öğrenin.",
    icon: <FileText className="h-8 w-8 text-chart-2" />,
    imageUrl: "https://images.pexels.com/photos/7412095/pexels-photo-7412095.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Doğal Dil İşleme (NLP), bilgisayarların insan (doğal) dilini anlama, yorumlama ve üretme yeteneği kazandıran yapay zeka disiplinidir. Metin ve konuşma verileri üzerinde çalışarak anlam çıkarma, duygu analizi yapma, diller arası çeviri gerçekleştirme, metin özetleme ve sohbet botları geliştirme gibi geniş bir uygulama yelpazesine sahiptir. **NLP süreci genellikle metin ön işleme (tokenizasyon, kök bulma, temizleme vb.), özellik çıkarımı ve model oluşturma adımlarını içerir.** Son yıllarda, **Transformer mimarisi ve Büyük Dil Modelleri (LLM'ler)** gibi tekniklerdeki ilerlemeler sayesinde NLP alanında çığır açan gelişmeler yaşanmıştır. Bu gelişmeler, duygu analizi, metin üretimi, soru yanıtlama ve makine çevirisi gibi daha önce mümkün olmayan karmaşık dil görevlerinin yerine getirilmesini sağlamıştır. NLP, insan ve bilgisayar arasındaki etkileşimi daha doğal ve akıcı hale getirerek birçok sektörde devrim yaratma potansiyeli taşımaktadır.",
    subtopics: [
      {
        title: "Metin Ön İşleme",
        description: "Metin verilerini analiz için hazırlama teknikleri (tokenizasyon, temizleme vb.).",
        imageUrl: "https://images.pexels.com/photos/267669/pexels-photo-267669.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/nlp/text-preprocessing"
      },
      {
        title: "Metin Analizi",
        description: "Metinleri işleme, temizleme ve yapılandırma teknikleri.",
        imageUrl: "https://images.pexels.com/photos/267669/pexels-photo-267669.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/nlp/text-analysis"
      },
      {
        title: "Dil Modelleri",
        description: "BERT, GPT ve diğer büyük dil modellerinin çalışma prensipleri.",
        imageUrl: "https://images.pexels.com/photos/1181271/pexels-photo-1181271.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/nlp/language-models"
      },
      {
        title: "Duygu Analizi",
        description: "Metinlerden duygu ve görüşleri çıkarma yöntemleri.",
        imageUrl: "https://images.pexels.com/photos/590022/pexels-photo-590022.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/nlp/sentiment-analysis"
      },
      {
        title: "Makine Çevirisi",
        description: "Diller arası otomatik çeviri sistemlerinin çalışma prensipleri.",
        imageUrl: "https://images.pexels.com/photos/267669/pexels-photo-267669.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/nlp/machine-translation"
      }
    ],
    skills: ["Metin Ön İşleme", "Vektör Temsilleri (Word Embeddings)", "Dil Modellemesi", "Transformer Mimarileri", "Duygu Analizi", "Metin Sınıflandırma", "Varlık Tanıma (NER)", "Metin Üretimi", "NLP Kütüphaneleri (NLTK, spaCy, Hugging Face)"],
    resources: [
      { title: "NLP Temelleri", type: "Kurs", link: "#" },
      { title: "Dil Modelleriyle Çalışma Atölyesi", type: "Atölye", link: "#" },
      { title: "Duygu Analizi Projesi Rehberi", type: "Pratik", link: "#" },
      { title: "Transformer Modellerine Giriş", type: "E-Kitap", link: "#" }
    ]
  },
  "computer-vision": {
    title: "Bilgisayarlı Görü",
    description: "Bilgisayarların görüntüleri nasıl algıladığını ve işlediğini anlayın.",
    icon: <Eye className="h-8 w-8 text-chart-3" />,
    imageUrl: "https://images.pexels.com/photos/8438922/pexels-photo-8438922.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Bilgisayarlı görü (Computer Vision), makinelerin dijital görüntü veya videolardan anlamlı bilgiler çıkarma yeteneği kazandıran bir yapay zeka alanıdır. İnsan gözünün görsel dünyayı algılama ve yorumlama sürecini taklit etmeyi hedefler. Görüntü sınıflandırma (bir görüntünün ne olduğunu belirleme), nesne algılama (görüntüdeki nesnelerin yerini ve türünü belirleme), görüntü segmentasyonu (görüntüyü farklı bölgelere ayırma), yüz tanıma ve hareket takibi gibi çok çeşitli görevleri kapsar. Bilgisayarlı görüdeki ilerlemeler, otonom araçlar, güvenlik sistemleri, tıbbi görüntü analizi, artırılmış gerçeklik ve endüstriyel otomasyon gibi birçok alanda devrimci uygulamaların önünü açmıştır. Konvolüsyonel Sinir Ağları (CNN'ler) bu alandaki en etkili derin öğrenme modellerindendir.",
    subtopics: [
      {
        title: "Görüntü Sınıflandırma",
        description: "Görüntüleri kategorilere ayırma teknikleri.",
        imageUrl: "https://images.pexels.com/photos/60504/security-protection-anti-virus-software-60504.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/computer-vision/image-classification"
      },
      {
        title: "Nesne Tespiti",
        description: "Görüntülerdeki nesneleri tespit etme ve konumlandırma.",
        imageUrl: "https://images.pexels.com/photos/762679/pexels-photo-762679.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/computer-vision/object-detection"
      },
      {
        title: "Yüz Tanıma",
        description: "Yüz tanıma sistemlerinin çalışma prensipleri ve uygulamaları.",
        imageUrl: "https://images.pexels.com/photos/6203795/pexels-photo-6203795.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/computer-vision/face-recognition"
      },
      {
        title: "Görüntü Segmentasyonu",
        description: "Görüntüleri anlamlı bölgelere ayırma teknikleri.",
        imageUrl: "https://images.pexels.com/photos/2599244/pexels-photo-2599244.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/computer-vision/image-segmentation"
      }
    ],
    skills: ["Görüntü İşleme Temelleri", "Konvolüsyonel Sinir Ağları (CNN)", "Nesne Algılama Algoritmaları (YOLO, Faster R-CNN)", "Görüntü Segmentasyonu Teknikleri", "Öznitelik Çıkarımı", "OpenCV", "PyTorch/TensorFlow ile Görüntü İşleme", "Model Eğitimi ve Değerlendirme"],
    resources: []
  },
  "generative-ai": {
    title: "Üretken AI",
    description: "Metin, görüntü ve ses üretebilen yapay zeka modellerini keşfedin.",
    icon: <Lightbulb className="h-8 w-8 text-chart-4" />,
    imageUrl: "https://images.pexels.com/photos/8386434/pexels-photo-8386434.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Üretken Yapay Zeka (Generative AI), mevcut verilerden öğrenerek yeni ve özgün içerikler (metin, görüntü, müzik, kod, video vb.) oluşturabilen yapay zeka modellerini tanımlar. Bu modeller, verilerdeki kalıpları, stilleri ve yapıları öğrenir ve bu bilgiyi kullanarak daha önce hiç görülmemiş çıktılar üretir. Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs) ve Diffusion modelleri gibi teknikler, üretken AI'nın temelini oluşturur. ChatGPT, DALL-E, Midjourney ve Stable Diffusion gibi popüler araçlar, üretken AI'nın potansiyelini sergileyen güncel örneklerdir. Yaratıcı endüstrilerden bilimsel araştırmalara, yazılım geliştirmeden eğitime kadar birçok alanda dönüştürücü etkilere sahiptir. Üretken AI, makinelerin sadece analiz etmekle kalmayıp aynı zamanda 'yaratıcılık' sergileyebileceği bir geleceğin kapılarını aralamaktadır.",
    subtopics: [
      {
        title: "Üretken Çekişmeli Ağlar (GAN)",
        description: "GAN'ların yapısı ve gerçekçi içerik üretme yöntemleri.",
        imageUrl: "https://images.pexels.com/photos/7567434/pexels-photo-7567434.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/generative-ai/gan"
      },
      {
        title: "Diffusion Modelleri",
        description: "Diffusion modellerinin çalışma prensipleri ve uygulamaları.",
        imageUrl: "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/generative-ai/diffusion-models"
      },
      {
        title: "Büyük Dil Modelleri",
        description: "Metin üreten yapay zeka modellerinin yapısı ve eğitimi.",
        imageUrl: "https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/generative-ai/large-language-models"
      },
      {
        title: "Text-to-Image Modelleri",
        description: "Metinden görüntü üreten modellerin çalışma prensipleri.",
        imageUrl: "https://images.pexels.com/photos/8566460/pexels-photo-8566460.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/generative-ai/text-to-image"
      }
    ],
    skills: ["Derin Öğrenme", "GANs", "VAEs", "Diffusion Modelleri", "Transformer Modelleri", "Metin/Görüntü/Ses Üretimi", "Model Eğitimi ve Fine-tuning", "Prompt Engineering", "Üretken AI Uygulama Geliştirme"],
    resources: [
      { title: "Üretken AI Temelleri", type: "Kurs", link: "#" },
      { title: "GAN ile Görüntü Üretimi Uygulamaları", type: "Pratik", link: "#" },
      { title: "Büyük Dil Modelleriyle (LLM) Çalışma Atölyesi", type: "Atölye", link: "#" },
      { title: "Diffusion Modellerine Kapsamlı Bakış", type: "E-Kitap", link: "#" }
    ]
  },
  "neural-networks": {
    title: "Sinir Ağları",
    description: "Beynin çalışma prensibinden esinlenen yapay sinir ağları hakkında bilgi edinin.",
    icon: <Brain className="h-8 w-8 text-chart-5" />,
    imageUrl: "https://images.pexels.com/photos/8386421/pexels-photo-8386421.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Yapay Sinir Ağları (YSA), insan beyninin yapısından ve işleyişinden esinlenerek tasarlanmış hesaplama modelleridir. Birbirine bağlı 'nöron' katmanlarından oluşurlar ve öğrenme süreci, bu bağlantıların ağırlıklarının ve eğilimlerinin ayarlanmasıyla gerçekleşir. YSA'lar, özellikle karmaşık örüntü tanıma, sınıflandırma, regresyon ve tahmin görevlerinde üstün başarı gösterirler. Derin Öğrenme, birden çok gizli katmana sahip sinir ağlarını ifade eder ve son yıllarda görüntü tanıma, doğal dil işleme ve konuşma tanıma gibi alanlarda büyük atılımlar sağlamıştır. Konvolüsyonel Sinir Ağları (CNN) ve Tekrarlayan Sinir Ağları (RNN) gibi özel sinir ağı mimarileri, farklı veri türleri (görüntü, metin, zaman serisi) için optimize edilmiştir. Sinir ağları, modern yapay zekanın temel taşlarından biridir ve öğrenme yetenekleri sayesinde birçok zorlu problemin çözümünde kritik rol oynar.",
    subtopics: [
      {
        title: "Temel Sinir Ağı Mimarileri",
        description: "Temel yapay sinir ağı yapıları ve çalışma prensipleri.",
        imageUrl: "https://images.pexels.com/photos/1181271/pexels-photo-1181271.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/neural-networks/temel-sinir-agi-mimarileri"
      },
      {
        title: "Yapay Sinir Ağları ve Güvenlik",
        description: "Yapay sinir ağlarının güvenlik alanındaki uygulamaları ve önemi.",
        imageUrl: "https://images.pexels.com/photos/5325710/pexels-photo-5325710.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/neural-networks/yapay-sinir-aglari-guvenlik-uygulamalari"
      },
      {
        title: "Konvolüsyonel Sinir Ağları (CNN)",
        description: "Görüntü işlemede kullanılan CNN'lerin yapısı ve uygulamaları.",
        imageUrl: "https://images.pexels.com/photos/60504/security-protection-anti-virus-software-60504.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/neural-networks/konvolusyonel-sinir-aglari"
      },
      {
        title: "Tekrarlayan Sinir Ağları (RNN)",
        description: "Dizileri işleyen RNN'lerin yapısı ve kullanım alanları.",
        imageUrl: "https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/neural-networks/tekrarlayan-sinir-aglari-rnn"
      },
      {
        title: "Transformerlar",
        description: "Modern NLP'nin temelini oluşturan transformer mimarisi.",
        imageUrl: "https://images.pexels.com/photos/267669/pexels-photo-267669.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/neural-networks/transformerlar"
      }
    ],
    skills: ["Nöron Modelleri", "İleri Beslemeli Ağlar", "Geri Yayılım (Backpropagation)", "Evrişimsel Katmanlar (Convolutional Layers)", "Tekrarlayan Katmanlar (Recurrent Layers)", "Aktivasyon Fonksiyonları", "Model Eğitimi", "Hiperparametre Ayarlama", "Derin Öğrenme Kütüphaneleri (TensorFlow, PyTorch)"],
    resources: [
      { title: "Sinir Ağları Temelleri", type: "Kurs", link: "#" },
      { title: "PyTorch ile Sinir Ağı Uygulamaları", type: "Pratik", link: "#" },
      { title: "Derin Öğrenme Mimarilerine Giriş", type: "E-Kitap", link: "#" },
      { title: "Sinir Ağı Optimizasyon Teknikleri", type: "Kaynak", link: "#" }
    ]
  },
  "ai-ethics": {
    title: "AI Etiği",
    description: "Yapay zekanın etik kullanımı ve toplumsal etkileri üzerine tartışmalar.",
    icon: <Users className="h-8 w-8 text-chart-1" />,
    imageUrl: "https://images.pexels.com/photos/8386422/pexels-photo-8386422.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Yapay Zeka Etiği, yapay zeka sistemlerinin geliştirilmesi, dağıtılması ve kullanılması sırasında ortaya çıkan ahlaki, sosyal ve felsefi soruları inceleyen disiplinler arası bir alandır. AI'ın hızla yaygınlaşmasıyla birlikte, şeffaflık, adalet, hesap verebilirlik, gizlilik ve güvenlik gibi konular kritik önem kazanmıştır. AI sistemlerindeki potansiyel önyargılar, işsizlik, otonom silahlar ve mahremiyetin ihlali gibi etik sorunlar, küresel düzeyde tartışılmaktadır. AI etiği, sadece teknik bir mesele olmayıp, aynı zamanda hukuki düzenlemeler, toplumsal normlar ve bireysel sorumlulukları da kapsar. Bu alandaki farkındalık ve çalışmalar, yapay zekanın insanlık için faydalı ve sorumlu bir şekilde geliştirilmesini ve kullanılmasını sağlamak amacıyla büyük önem taşımaktadır.",
    subtopics: [],
    skills: ["Etik Karar Verme Çerçeveleri", "AI Önyargısını Anlama ve Azaltma", "Şeffaflık ve Açıklanabilirlik (Explainable AI)", "Mahremiyet ve Veri Koruma", "AI Hukuku ve Regülasyonları", "Toplumsal Etki Analizi", "Hesap Verebilirlik Mekanizmaları"],
    resources: [
      { title: "AI Etiği Temelleri", type: "Kurs", link: "#" },
      { title: "Yapay Zeka ve Toplum", type: "E-Kitap", link: "#" },
      { title: "Sorumlu AI Geliştirme", type: "Rehber", link: "#" }
    ]
  },
  "metasezgisel-optimizasyon": {
    title: "Metasezgisel Optimizasyon",
    description: "Karmaşık optimizasyon problemlerini çözmek için doğadan esinlenen ve sezgisel yöntemler kullanan algoritmaları öğrenin.",
    icon: <Sigma className="h-8 w-8 text-purple-500" />,
    imageUrl: "/images/metasezgisel_algoritm.jpg",
    longDescription: "Metasezgisel optimizasyon algoritmaları, geleneksel optimizasyon yöntemlerinin yetersiz kaldığı büyük ve karmaşık problemler için güçlü çözüm yaklaşımları sunar. Bu algoritmalar genellikle doğadaki süreçlerden (evrim, sürü davranışları vb.) veya fiziksel olaylardan (metal tavlaması gibi) ilham alır. Kesin en iyi çözümü garanti etmeseler de, kabul edilebilir sürede çok iyi çözümler bulma konusunda etkilidirler. Bu bölümde, en yaygın metasezgisel algoritmaların çalışma prensiplerini, avantajlarını, dezavantajlarını ve çeşitli alanlardaki uygulamalarını inceleyeceğiz.",
    subtopics: [
      {
        title: "Genetik Algoritmalar",
        description: "Evrimsel süreçlerden ilham alan, popülasyon tabanlı bir optimizasyon tekniği.",
        imageUrl: "/images/genetic_algoritm.jpg", 
        href: "/topics/metasezgisel-optimizasyon/genetik-algoritmalar"
      },
      {
        title: "Parçacık Sürü Optimizasyonu",
        description: "Kuş sürülerinin ve balık okullarının sosyal davranışlarından esinlenen bir yöntem.",
        imageUrl: "https://images.pexels.com/photos/1089842/pexels-photo-1089842.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/metasezgisel-optimizasyon/parcacik-suru-optimizasyonu"
      },
      {
        title: "Tavlama Benzetimi",
        description: "Metal tavlama sürecinden esinlenen, olasılıksal bir arama tekniği.",
        imageUrl: "https://images.pexels.com/photos/162491/foundry-molten-metal-production-162491.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/metasezgisel-optimizasyon/tavlama-benzetimi"
      },
      {
        title: "Yasaklı Arama",
        description: "Arama sürecinde daha önce ziyaret edilen çözümlerin tekrarını engelleyen bir yöntem.",
        imageUrl: "https://images.pexels.com/photos/268917/pexels-photo-268917.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/metasezgisel-optimizasyon/yasakli-arama"
      },
      {
        title: "Uyum Araması",
        description: "Müzisyenlerin doğaçlama performanslarındaki estetik standartlardan esinlenir.",
        imageUrl: "https://images.pexels.com/photos/3783124/pexels-photo-3783124.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/metasezgisel-optimizasyon/uyum-aramasi"
      },
      {
        title: "Diferansiyel Gelişim",
        description: "Popülasyondaki çözümler arasındaki fark vektörlerini kullanarak yeni çözümler üretir.",
        imageUrl: "https://images.pexels.com/photos/1181275/pexels-photo-1181275.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/metasezgisel-optimizasyon/diferansiyel-gelisim"
      },
      {
        title: "Karınca Koloni Optimizasyonu",
        description: "Karıncaların yiyecek arama davranışlarından esinlenen olasılıksal bir algoritma.",
        imageUrl: "https://images.unsplash.com/photo-1588534496794-71d4493ea504?q=80&w=1287&auto=format&fit=crop", // Original: https://unsplash.com/photos/Vf1JrKMUS0Q
        href: "/topics/metasezgisel-optimizasyon/karinca-koloni-optimizasyonu"
      },
      {
        title: "Yapay Arı Kolonisi Optimizasyonu",
        description: "Bal arılarının yiyecek arama davranışlarından esinlenen bir algoritma.",
        imageUrl: "https://images.pexels.com/photos/790357/pexels-photo-790357.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2", // Generic bees/honeycomb
        href: "/topics/metasezgisel-optimizasyon/yapay-ari-kolonisi-optimizasyonu"
      },
      {
        title: "Ateşböceği Algoritması",
        description: "Ateşböceklerinin ışık sinyallerinden ve çekiciliklerinden esinlenir.",
        imageUrl: "https://images.unsplash.com/photo-1530053999930-791076eb2516?q=80&w=1470&auto=format&fit=crop", // Original: https://unsplash.com/s/photos/firefly
        href: "/topics/metasezgisel-optimizasyon/atesbocegi-algoritmasi"
      },
      {
        title: "Guguk Kuşu Araması",
        description: "Guguk kuşlarının kuluçka parazitizmi ve Lévy uçuşlarından esinlenir.",
        imageUrl: "https://images.unsplash.com/photo-1602601394064-069135805672?q=80&w=1374&auto=format&fit=crop", // Original: https://unsplash.com/photos/HlLvAaHU3H4 (Greater Coucal, a type of cuckoo)
        href: "/topics/metasezgisel-optimizasyon/guguk-kusu-aramasi"
      },
      {
        title: "Gri Kurt Optimizasyonu",
        description: "Gri kurtların sosyal hiyerarşisi ve avlanma davranışlarından esinlenir.",
        imageUrl: "https://images.pexels.com/photos/162318/wolf-gray-wolf-canis-lupus-predator-162318.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        href: "/topics/metasezgisel-optimizasyon/gri-kurt-optimizasyonu"
      },
      {
        title: "Balina Optimizasyon Algoritması",
        description: "Kambur balinaların kabarcık ağı avlanma tekniğinden esinlenir.",
        imageUrl: "https://images.pexels.com/photos/96423/pexels-photo-96423.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2", // Humpback whale
        href: "/topics/metasezgisel-optimizasyon/balina-optimizasyon-algoritmasi"
      },
      {
        title: "Yarasa Algoritması",
        description: "Yarasaların ekolokasyon davranışlarından esinlenir.",
        imageUrl: "https://images.pexels.com/photos/4754651/pexels-photo-4754651.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2", // Bat in flight
        href: "/topics/metasezgisel-optimizasyon/yarasa-algoritmasi"
      }
    ],
    skills: ["Problem Modelleme", "Algoritma Tasarımı", "Python ile Uygulama", "Performans Analizi", "Parametre Ayarlama", "Optimizasyon Temelleri", "Sezgisel Yöntemler"],
    resources: []
  },
  "responsible-ai": {
    title: "Sorumlu AI",
    description: "AI'nin etik ve sosyal sorumluluklarını anlayın.",
    icon: <Shield className="h-8 w-8 text-chart-1" />,
    imageUrl: "https://images.pexels.com/photos/8386422/pexels-photo-8386422.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    longDescription: "Sorumlu AI, AI'nin günümüzün ve geleceğin sosyal, ekonomik ve teknolojik dünyasında etik ve sosyal sorumluluklarını anlamanın önemini vurgular. Bu konu, AI'nin nasıl geliştirileceği, kullanılacağı ve yönetileceği konuları içerir. Sorumlu AI, AI'nin insanlarla etkileşimini ve onların haklarını koruma konusunda önemli rol oynar. Bu alandaki farkındalık ve çalışmalar, AI'nin insanlık için faydalı ve sorumlu bir şekilde geliştirilmesini ve kullanılmasını sağlamak amacıyla büyük önem taşımaktadır.",
    subtopics: [],
    skills: ["AI'nin Sosyal ve Ekonomik Sorumluluklarını Anlama", "AI'nin İnsan Haklarına Etkilerini İnceleme", "AI'nin İnsanlarla Etkileşimi İyileştirme", "AI'nin İnsanlık İçin Faydalı Olmasını Sağlama"],
    resources: [
      { title: "Sorumlu AI Temelleri", type: "Kurs", link: "#" },
      { title: "Sorumlu AI Pratikleri", type: "E-Kitap", link: "#" },
      { title: "Sorumlu AI Geliştirme", type: "Rehber", link: "#" }
    ]
  },
  "python": {
    title: "Python",
    description: "Python programlama dili ile ilgili temel ve ileri konuları keşfedin.",
    icon: <Code2 className="h-8 w-8 text-blue-500" />,
    imageUrl: "/images/python.jpg", // public/images/ altında bir python görseli ekleyin
    longDescription: "Python, hem başlangıç seviyesinde hem de ileri düzeyde yazılım geliştirme için kullanılan, güçlü ve çok yönlü bir programlama dilidir. Bu bölümde Python ile nesneye yönelik programlama ve derin öğrenme konularını bulabilirsiniz.",
    subtopics: [
      {
        title: "Python ile Nesneye Yönelik Programlama",
        description: "Sınıflar, nesneler, kalıtım ve daha fazlası ile Python'da OOP temelleri.",
        imageUrl: "/images/python_oop.jpg",
        href: "/topics/python/nesneye-yonelik-programlama"
      },
      {
        title: "Python ile Derin Öğrenme",
        description: "Python ile derin öğrenme temelleri ve popüler kütüphaneler.",
        imageUrl: "/images/python_deep_learning.jpg",
        href: "/topics/python/derin-ogrenme"
      }
    ],
    skills: ["Python Temelleri", "OOP", "Derin Öğrenme", "Kütüphaneler"],
    resources: []
  }
};

// List of all topics for related topics section
const allTopicSlugs = [
  "machine-learning", 
  "nlp", 
  "computer-vision", 
  "generative-ai", 
  "neural-networks", 
  "ai-ethics",
  "metasezgisel-optimizasyon",
  "responsible-ai",
  "python"
];

// Generate metadata for each topic page
export async function generateMetadata({ params }: { params: { slug: string } }): Promise<Metadata> {
  const { slug } = params;
  const topic = topicsData[slug] || {
    title: "Konu Bulunamadı",
    description: "İstediğiniz konu şu anda mevcut değil.",
  };
  
  return {
    title: `${topic.title} Eğitimi | Kodleon Yapay Zeka Platformu`,
    description: `${topic.description} Kodleon'da ${topic.title.toLowerCase()} konusunu derinlemesine öğrenin.`,
    keywords: `${topic.title.toLowerCase()}, ${topic.title.toLowerCase()} eğitimi, kodleon, yapay zeka, AI öğrenme, türkçe ${topic.title.toLowerCase()} kursu`,
    alternates: {
      canonical: `https://kodleon.com/topics/${slug}`,
    },
    openGraph: {
      title: `${topic.title} Eğitimi | Kodleon`,
      description: topic.description,
      url: `https://kodleon.com/topics/${slug}`,
      images: [
        {
          url: topic.imageUrl,
          width: 1200,
          height: 630,
          alt: `${topic.title} - Kodleon yapay zeka eğitimi`,
        }
      ],
    },
  };
}

// Add generateStaticParams function
export async function generateStaticParams() {
  return allTopicSlugs.map((slug) => ({
    slug: slug,
  }));
}

// 1. longDescription görselleştirme
const highlightKeywords = (text: string) => {
  return text
    .replace(/Yapay Sinir Ağları \(YSA\)/g, '<span class="font-bold text-primary">Yapay Sinir Ağları (YSA)</span>')
    .replace(/Derin Öğrenme/g, '<span class="font-bold text-pink-600 dark:text-pink-400">Derin Öğrenme</span>')
    .replace(/CNN/g, '<span class="font-bold text-blue-600 dark:text-blue-400">CNN</span>')
    .replace(/RNN/g, '<span class="font-bold text-green-600 dark:text-green-400">RNN</span>');
};

// 2. Subtopic ikon eşlemesi
const subtopicIcons: Record<string, JSX.Element> = {
  'Temel Sinir Ağı Mimarileri': <GraduationCap className="h-5 w-5 text-primary" />, 
  'Yapay Sinir Ağları ve Güvenlik': <Shield className="h-5 w-5 text-amber-600" />, 
  'Konvolüsyonel Sinir Ağları (CNN)': <Shapes className="h-5 w-5 text-blue-600" />, 
  'Tekrarlayan Sinir Ağları (RNN)': <BrainCircuit className="h-5 w-5 text-green-600" />, 
  'Transformerlar': <GitMerge className="h-5 w-5 text-fuchsia-600" />
};

// 3. Skill ikon eşlemesi
const skillIcons: Record<string, JSX.Element> = {
  'Nöron Modelleri': <Brain className="h-4 w-4 mr-1 text-primary" />,
  'İleri Beslemeli Ağlar': <ArrowRight className="h-4 w-4 mr-1 text-blue-600" />,
  'Geri Yayılım (Backpropagation)': <ArrowLeft className="h-4 w-4 mr-1 text-pink-600" />,
  'Evrişimsel Katmanlar (Convolutional Layers)': <Shapes className="h-4 w-4 mr-1 text-blue-600" />,
  'Tekrarlayan Katmanlar (Recurrent Layers)': <BrainCircuit className="h-4 w-4 mr-1 text-green-600" />,
  'Aktivasyon Fonksiyonları': <Sigma className="h-4 w-4 mr-1 text-fuchsia-600" />,
  'Model Eğitimi': <GraduationCap className="h-4 w-4 mr-1 text-amber-600" />,
  'Hiperparametre Ayarlama': <Settings2 className="h-4 w-4 mr-1 text-cyan-600" />,
  'Derin Öğrenme Kütüphaneleri (TensorFlow, PyTorch)': <Code2 className="h-4 w-4 mr-1 text-orange-600" />
};

export default function TopicPage({ params }: { params: { slug: string } }) {
  const { slug } = params;
  const topic = topicsData[slug] || {
    title: "Konu Bulunamadı",
    description: "İstediğiniz konu şu anda mevcut değil.",
    imageUrl: "https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    subtopics: [],
    skills: [],
    resources: []
  };
  
  // Get 3 related topics (excluding current one)
  const relatedTopicSlugs = allTopicSlugs
    .filter(s => s !== slug)
    .sort(() => 0.5 - Math.random())
    .slice(0, 3);
  
  const relatedTopics = relatedTopicSlugs.map(s => ({
    slug: s,
    ...topicsData[s]
  }));

  return (
    <div>
      {/* Hero section */}
      <section className="relative" aria-labelledby="topic-title">
        <div className="relative h-[300px] md:h-[400px]">
          <Image 
            src={topic.imageUrl}
            alt={`${topic.title} eğitimi - Kodleon yapay zeka platformu`}
            fill
            className="object-cover"
            priority={true}
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/80 to-transparent" />
        </div>
        <div className="container max-w-6xl mx-auto relative -mt-32 pb-12">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 mb-4">
              <Button asChild variant="ghost" size="sm" className="gap-1">
                <Link href="/topics" aria-label="Tüm yapay zeka konularına dön">
                  <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                  Tüm Konular
                </Link>
              </Button>
            </div>
            <div className="flex items-center gap-4 mb-6">
              <div className="p-3 rounded-full bg-primary/10 backdrop-blur-sm">
                {topic.icon}
              </div>
              <h1 id="topic-title" className="text-4xl font-bold">{topic.title}</h1>
            </div>
            <p className="text-xl text-muted-foreground">
              {topic.description}
            </p>
          </div>
        </div>
      </section>
      
      {/* Main content */}
      <section className="container max-w-6xl mx-auto py-12" aria-labelledby="overview-heading">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 max-w-6xl mx-auto">
          <div>
            <div className="prose prose-lg dark:prose-invert max-w-none">
              <h2 id="overview-heading" className="text-3xl font-bold text-primary dark:text-primary-dark mb-4 mt-8">Genel Bakış</h2>
              <div className="rounded-lg bg-primary/10 p-4 mb-4 border border-primary/20">
                {topic.longDescription ? <MarkdownContent content={topic.longDescription} /> : <p>Açıklama bulunamadı.</p>}
              </div>
              
              {topic.subtopics && topic.subtopics.length > 0 && (
                <>
                  <h2 id="subtopics-heading" className="text-3xl font-bold text-primary dark:text-primary-dark mb-4 mt-8">Alt Konular</h2>
                  {slug === "metasezgisel-optimizasyon" ? (
                    <PaginatedSubtopicsList
                      subtopics={topic.subtopics}
                      isMetasearchTopic={true}
                      subtopicIcons={subtopicIcons}
                      itemsPerPage={8}
                    />
                  ) : (
                  <TopicSubtopicsList 
                    subtopics={topic.subtopics}
                    isMetasearchTopic={slug === "metasezgisel-optimizasyon"}
                    subtopicIcons={subtopicIcons}
                  />
                  )}
                </>
              )}
            </div>

            {/* Learning Journey Section - Specific to Machine Learning Topic */}
            {slug === "machine-learning" && (
              <div className="mt-12 pt-8 border-t border-border">
                <h2 id="learning-journey-heading" className="text-3xl font-bold mb-8 text-center">Makine Öğrenmesi Yolculuğunuz</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {[{
                    title: "Temel Matematik ve İstatistik",
                    description: "Lineer cebir, kalkülüs ve olasılık gibi temel matematiksel kavramlar ile istatistiksel analiz ve hipotez testi temelleri.",
                    icon: <Sigma className="w-10 h-10 text-blue-500" />
                  }, {
                    title: "Python Programlama",
                    description: "Veri yapıları, kontrol akışı, fonksiyonlar ve NumPy, Pandas gibi temel veri bilimi kütüphaneleri.",
                    icon: <Code2 className="w-10 h-10 text-green-500" />
                  }, {
                    title: "Veri Analizi ve Görselleştirme",
                    description: "Veri temizleme, dönüştürme, keşifsel veri analizi (EDA) ve Matplotlib, Seaborn gibi araçlarla etkili görselleştirmeler.",
                    icon: <BarChart3 className="w-10 h-10 text-yellow-500" />
                  }, {
                    title: "Temel ML Algoritmaları",
                    description: "Regresyon, sınıflandırma, kümeleme gibi temel algoritma türlerini ve çalışma prensiplerini anlama.",
                    icon: <BrainCircuit className="w-10 h-10 text-purple-500" />
                  }, {
                    title: "Model Geliştirme ve Değerlendirme",
                    description: "Veri bölme, model eğitimi, hiperparametre ayarı, çapraz doğrulama ve performans metrikleri ile model değerlendirme.",
                    icon: <ClipboardCheck className="w-10 h-10 text-red-500" />
                  }, {
                    title: "İleri Düzey Konular ve Uzmanlaşma",
                    description: "Derin öğrenme, doğal dil işleme, bilgisayarlı görü gibi alanlarda uzmanlaşma veya MLOps gibi konulara yönelme.",
                    icon: <GraduationCap className="w-10 h-10 text-indigo-500" />
                  }].map((step, index) => (
                    <Card key={index} className="flex flex-col items-center p-6 text-center hover:shadow-lg transition-shadow">
                      <div className="p-3 bg-primary/10 rounded-full mb-4">
                        {step.icon}
                      </div>
                      <CardTitle className="text-xl mb-2">{step.title}</CardTitle>
                      <CardDescription>{step.description}</CardDescription>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          <div>
            <div className="bg-muted rounded-lg p-6 sticky top-24 max-w-sm mx-auto">
              <h3 className="text-xl font-semibold mb-4">Bu Konuda Kazanacağınız Beceriler</h3>
              {topic.skills && topic.skills.length > 0 && (
                <ul className="space-y-3 mb-6">
                  {topic.skills?.map((skill: string, index: number) => (
                    <li key={index} className="flex items-start gap-2">
                      <CheckCircle2 className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" aria-hidden="true" />
                      <span>{skill}</span>
                    </li>
                  ))}
                </ul>
              )}
              
              {topic.resources && topic.resources.length > 0 && (
                <>
                  <Separator className="my-6" />
                
                  <h3 className="text-xl font-semibold mb-4">
                    {slug === "metasezgisel-optimizasyon" ? "Diğer Algoritmalar" : "Önerilen Kaynaklar"}
                  </h3>
                  <ul className="space-y-4">
                    {topic.resources?.map((resource: any, index: number) => (
                      <li key={index}>
                        <Link 
                          href={resource.href || resource.link || '#'}
                          className="flex items-center justify-between p-3 bg-background rounded-md hover:bg-secondary transition-colors"
                          aria-label={`${resource.title} kaynağını incele - ${resource.type || ''}`}
                        >
                          <span className="font-medium">{resource.title}</span>
                          {resource.type && <span className="text-sm text-muted-foreground">{resource.type}</span>}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </>
              )}

              {slug === "machine-learning" && (
                <>
                  <Separator className="my-6" />
                  <h3 className="text-xl font-semibold mb-4">Makine Öğrenmesi Simülatörü</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Makine öğrenmesi algoritmalarını interaktif bir ortamda deneyimleyin ve temel kavramları uygulamalı olarak pekiştirin.
                  </p>
                  <Button asChild className="w-full bg-green-600 hover:bg-green-700">
                    <Link href="https://ml.kodleon.com" target="_blank" rel="noopener noreferrer">
                      <FlaskConical className="mr-2 h-4 w-4" /> Simülatöre Git
                    </Link>
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      </section>
      
      {/* Related topics */}
      <section className="bg-muted py-16" aria-labelledby="related-topics-heading">
        <div className="container max-w-6xl mx-auto">
          <h2 id="related-topics-heading" className="text-2xl font-bold mb-8">İlgili Konular</h2>
          {relatedTopics && relatedTopics.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {relatedTopics.map((relatedTopic, index) => (
                <Card key={index} className="overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1">
                  <div className="relative h-48">
                    <Image 
                      src={relatedTopic.imageUrl}
                      alt={`${relatedTopic.title} eğitimi - İlgili yapay zeka konusu`}
                      fill
                      className="object-cover"
                      loading="lazy"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
                    <div className="absolute bottom-4 left-4 p-2 rounded-full bg-background/80 backdrop-blur-sm">
                      {relatedTopic.icon}
                    </div>
                  </div>
                  <CardHeader>
                    <CardTitle>{relatedTopic.title}</CardTitle>
                    <CardDescription>{relatedTopic.description}</CardDescription>
                  </CardHeader>
                  <CardFooter>
                    <Button asChild variant="ghost" className="gap-1 ml-auto">
                      <Link href={`/topics/${relatedTopic.slug}`} aria-label={`${relatedTopic.title} konusunu keşfedin`}>
                        Konuyu İncele
                        <ArrowRight className="h-4 w-4" aria-hidden="true" />
                      </Link>
                    </Button>
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Structured data for SEO */}
      <script 
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "Course",
            "name": `${topic.title} Eğitimi`,
            "description": topic.description,
            "provider": {
              "@type": "Organization",
              "name": "Kodleon",
              "sameAs": "https://kodleon.com"
            },
            "courseCode": slug,
            "educationalLevel": "Beginner to Advanced",
            "teaches": topic.skills?.join(", "),
            "hasCourseInstance": {
              "@type": "CourseInstance",
              "courseMode": "online",
              "inLanguage": "tr"
            }
          })
        }}
      />
    </div>
  );
}