'use client';

import Link from "next/link";
import Image from "next/image";
import {
  ArrowRight, Brain, Database, Eye, FileText, Lightbulb, Shapes, Users, Zap, Award, UsersRound, BookCopy, Target as TargetIcon, Briefcase, MessageSquareHeart, Code2, Rss
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import FeaturedBlogCarousel from "@/components/FeaturedBlogCarousel";
import { LanguageSwitcher } from '@/components/LanguageSwitcher';
import { useTranslation } from '@/lib/i18n';
// import { TranslatedContent } from '@/components/TranslatedContent'; // This might be redundant now

export default function HomePageClientContent() {
  const { t, locale } = useTranslation();

  const topics = [
    {
      title: t('home.topics.items.machineLearning.title'),
      description: t('home.topics.items.machineLearning.description'),
      icon: <Database className="h-8 w-8 text-chart-1" aria-hidden="true" />,
      href: "/topics/machine-learning",
      imageUrl: "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
    },
    {
      title: t('home.topics.items.nlp.title'),
      description: t('home.topics.items.nlp.description'),
      icon: <FileText className="h-8 w-8 text-chart-2" aria-hidden="true" />,
      href: "/topics/nlp",
      imageUrl: "https://images.pexels.com/photos/7412095/pexels-photo-7412095.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
    },
    {
      title: t('home.topics.items.computerVision.title'),
      description: t('home.topics.items.computerVision.description'),
      icon: <Eye className="h-8 w-8 text-chart-3" aria-hidden="true" />,
      href: "/topics/computer-vision",
      imageUrl: "https://images.pexels.com/photos/8438922/pexels-photo-8438922.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
    },
    {
      title: t('home.topics.items.generativeAi.title'),
      description: t('home.topics.items.generativeAi.description'),
      icon: <Lightbulb className="h-8 w-8 text-chart-4" aria-hidden="true" />,
      href: "/topics/generative-ai",
      imageUrl: "https://images.pexels.com/photos/8386434/pexels-photo-8386434.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
    },
    {
      title: t('home.topics.items.neuralNetworks.title'),
      description: t('home.topics.items.neuralNetworks.description'),
      icon: <Brain className="h-8 w-8 text-chart-5" aria-hidden="true" />,
      href: "/topics/neural-networks",
      imageUrl: "https://images.pexels.com/photos/8386421/pexels-photo-8386421.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
    },
    {
      title: t('home.topics.items.aiEthics.title'),
      description: t('home.topics.items.aiEthics.description'),
      icon: <Users className="h-8 w-8 text-chart-1" aria-hidden="true" />,
      href: "/topics/ai-ethics",
      imageUrl: "https://images.pexels.com/photos/8386422/pexels-photo-8386422.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
    },
  ];

  const whyKodleonFeatures = [
    {
      title: t('home.whyKodleon.features.expertInstructors.title'),
      description: t('home.whyKodleon.features.expertInstructors.description'),
      icon: <Award className="h-10 w-10 text-primary" />
    },
    {
      title: t('home.whyKodleon.features.comprehensiveContent.title'),
      description: t('home.whyKodleon.features.comprehensiveContent.description'),
      icon: <BookCopy className="h-10 w-10 text-primary" />
    },
    {
      title: t('home.whyKodleon.features.handsOnProjects.title'),
      description: t('home.whyKodleon.features.handsOnProjects.description'),
      icon: <Briefcase className="h-10 w-10 text-primary" />
    },
    {
      title: t('home.whyKodleon.features.activeCommunity.title'),
      description: t('home.whyKodleon.features.activeCommunity.description'),
      icon: <UsersRound className="h-10 w-10 text-primary" />
    },
    {
      title: t('home.whyKodleon.features.turkishResources.title'),
      description: t('home.whyKodleon.features.turkishResources.description'),
      icon: <MessageSquareHeart className="h-10 w-10 text-primary" />
    },
    {
      title: t('home.whyKodleon.features.flexibleLearning.title'),
      description: t('home.whyKodleon.features.flexibleLearning.description'),
      icon: <Zap className="h-10 w-10 text-primary" />
    }
  ];

  const whatYouCanAchieve = [
    {
      title: t('home.achieve.items.smartApps.title'),
      description: t('home.achieve.items.smartApps.description'),
      icon: <Lightbulb className="h-8 w-8 text-green-500" />
    },
    {
      title: t('home.achieve.items.createValue.title'),
      description: t('home.achieve.items.createValue.description'),
      icon: <Database className="h-8 w-8 text-blue-500" />
    },
    {
      title: t('home.achieve.items.automationSolutions.title'),
      description: t('home.achieve.items.automationSolutions.description'),
      icon: <Zap className="h-8 w-8 text-purple-500" />
    },
    {
      title: t('home.achieve.items.leadTechnologies.title'),
      description: t('home.achieve.items.leadTechnologies.description'),
      icon: <TargetIcon className="h-8 w-8 text-red-500" />
    }
  ];

  const latestBlogPosts = [
    {
      title: "2025'in En Güçlü AI Modelleri: Öğrenciler İçin Ücretsiz Seçenekler",
      snippet: "2025 yılı ortalarında yapay zeka modellerinin son durumu, ücretsiz kullanım seçenekleri ve öğrenciler için özel fırsatlar hakkında kapsamlı rehber.",
      imageUrl: "/blog-images/ai-future-2025.jpg",
      href: "/blog/guncel-ai-modelleri-2025",
      date: new Date(2025, 5, 1).toLocaleDateString(locale, { year: 'numeric', month: 'long', day: 'numeric' }),
      category: t('home.blog.categories.aiDevelopments')
    },
    {
      title: t('home.blog.posts.embodiedAi.title'),
      snippet: t('home.blog.posts.embodiedAi.snippet'),
      imageUrl: "https://images.pexels.com/photos/7661169/pexels-photo-7661169.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
      href: "/blog/embodied-ai-future",
      date: new Date(2025, 4, 31).toLocaleDateString(locale, { year: 'numeric', month: 'long', day: 'numeric' }),
      category: t('home.blog.categories.aiDevelopments')
    },
    {
      title: t('home.blog.posts.aiCodeAssistants.title'),
      snippet: t('home.blog.posts.aiCodeAssistants.snippet'),
      imageUrl: "https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
      href: "/blog/ai-kod-asistanlari-karsilastirmasi",
      date: new Date(2025, 4, 31).toLocaleDateString(locale, { year: 'numeric', month: 'long', day: 'numeric' }),
      category: t('home.blog.categories.aiTools')
    }
  ];

  return (
    <div className="flex flex-col bg-background text-foreground">
      {/* Hero section */}
      <section className="relative py-16 md:py-24" aria-labelledby="hero-heading">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-secondary/10 dark:from-primary/5 dark:via-transparent dark:to-secondary/5" />
        <div 
          className="absolute inset-0 opacity-20 dark:opacity-10"
          style={{
            backgroundImage: 'url("data:image/svg+xml,%3Csvg width="30" height="30" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="%239C92AC" fill-opacity="0.1"%3E%3Cpath d="M0 10 L10 0 L20 10 L10 20 Z"/%3E%3C/g%3E%3C/svg%3E")',
            backgroundSize: '30px 30px'
          }}
          aria-hidden="true"
        />
        <div className="container max-w-6xl mx-auto relative z-10 px-4">
          <div className="max-w-3xl mx-auto text-center mb-8 md:mb-10">
            <div className="inline-block px-4 py-1 rounded-full bg-primary/10 text-primary font-medium text-sm mb-4">
              YENİ! 2025 GÜNCEL AI MODELLERİ REHBERİ
            </div>
            <h1 
              id="hero-heading" 
              className="text-3xl md:text-5xl font-bold tracking-tight mb-4 
                         bg-clip-text text-transparent bg-gradient-to-r from-primary via-pink-500 to-orange-500 
                         animate-gradient-xy"
            >
              {t('home.hero.title')}
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground mb-8">
              {t('home.hero.subtitle')} <span className="font-medium text-primary">Güncel AI modelleri ve öğrenciler için ücretsiz seçenekler hakkındaki rehberimizi keşfedin!</span>
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild size="lg" className="rounded-full shadow-lg hover:shadow-xl transition-shadow">
                <Link href="/blog/guncel-ai-modelleri-2025">
                  AI Modelleri Rehberini İncele
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="rounded-full border-border hover:border-primary/70 transition-colors">
                <Link href="/topics">
                  {t('home.hero.cta')}
                </Link>
              </Button>
            </div>
          </div>

          <FeaturedBlogCarousel posts={latestBlogPosts} />
        </div>
      </section>
      
      {/* Featured Learning Paths - NEW SECTION */}
      <section className="relative z-20 py-12 bg-gradient-to-r from-primary/10 via-background to-secondary/10">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-8">
            <span className="inline-block px-4 py-1 rounded-full bg-primary/10 text-primary font-medium text-sm mb-3">
              {t('home.featuredPaths.badge')}
            </span>
            <h2 className="text-2xl md:text-3xl font-bold tracking-tight mb-3">
              {t('home.featuredPaths.title')}
            </h2>
            <p className="text-base text-muted-foreground max-w-2xl mx-auto mb-6">
              {t('home.featuredPaths.subtitle')}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Python Learning Path */}
            <div className="relative overflow-hidden rounded-xl border border-primary/20 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950/30 dark:to-blue-900/20 shadow-lg hover:shadow-xl transition-all duration-300 group">
              <div className="absolute top-0 right-0 w-32 h-32 -mt-8 -mr-8 bg-blue-500/20 rounded-full blur-2xl"></div>
              <div className="p-6 md:p-8 relative z-10">
                <div className="flex items-start gap-4 mb-4">
                  <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/30">
                    <Code2 className="h-8 w-8 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <h3 className="text-xl md:text-2xl font-bold text-blue-800 dark:text-blue-300 mb-2">Python Öğrenme Yolu</h3>
                    <p className="text-sm md:text-base text-blue-700/80 dark:text-blue-300/80">Programlamanın temellerinden ileri Python uygulamalarına kadar kapsamlı eğitim içeriği</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 mb-6">
                  <div className="flex items-center gap-2 text-sm text-blue-700 dark:text-blue-300/90">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500"></div>
                    <span>Temel Python</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-blue-700 dark:text-blue-300/90">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500"></div>
                    <span>Veri Yapıları</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-blue-700 dark:text-blue-300/90">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500"></div>
                    <span>OOP</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-blue-700 dark:text-blue-300/90">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500"></div>
                    <span>Veri Analizi</span>
                  </div>
                </div>
                <Button asChild className="w-full bg-blue-600 hover:bg-blue-700 text-white border-none">
                  <Link href="/topics/python">
                    Python Derslerine Başla
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>

            {/* AI Learning Path */}
            <div className="relative overflow-hidden rounded-xl border border-primary/20 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950/30 dark:to-purple-900/20 shadow-lg hover:shadow-xl transition-all duration-300 group">
              <div className="absolute top-0 right-0 w-32 h-32 -mt-8 -mr-8 bg-purple-500/20 rounded-full blur-2xl"></div>
              <div className="p-6 md:p-8 relative z-10">
                <div className="flex items-start gap-4 mb-4">
                  <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/30">
                    <Brain className="h-8 w-8 text-purple-600 dark:text-purple-400" />
                  </div>
                  <div>
                    <h3 className="text-xl md:text-2xl font-bold text-purple-800 dark:text-purple-300 mb-2">Yapay Zeka Öğrenme Yolu</h3>
                    <p className="text-sm md:text-base text-purple-700/80 dark:text-purple-300/80">Yapay zeka temellerinden ileri uygulamalara kadar kapsamlı eğitim içeriği</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 mb-6">
                  <div className="flex items-center gap-2 text-sm text-purple-700 dark:text-purple-300/90">
                    <div className="w-1.5 h-1.5 rounded-full bg-purple-500"></div>
                    <span>Makine Öğrenmesi</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-purple-700 dark:text-purple-300/90">
                    <div className="w-1.5 h-1.5 rounded-full bg-purple-500"></div>
                    <span>Derin Öğrenme</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-purple-700 dark:text-purple-300/90">
                    <div className="w-1.5 h-1.5 rounded-full bg-purple-500"></div>
                    <span>NLP</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-purple-700 dark:text-purple-300/90">
                    <div className="w-1.5 h-1.5 rounded-full bg-purple-500"></div>
                    <span>Bilgisayarlı Görü</span>
                  </div>
                </div>
                <Button asChild className="w-full bg-purple-600 hover:bg-purple-700 text-white border-none">
                  <Link href="/topics/ai-fundamentals">
                    Yapay Zeka Derslerine Başla
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Free AI Courses section */}
      <section className="relative z-20 py-16 bg-background">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-10">
            <span className="inline-block px-4 py-1 rounded-full bg-primary/10 text-primary font-medium text-sm mb-4">
              {t('home.freeCourses.badge')}
            </span>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-4">
              {t('home.freeCourses.title')}
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
              {t('home.freeCourses.subtitle')}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
            <Card className="group hover:shadow-lg transition-all duration-300 border-blue-200 dark:border-blue-900/40 bg-gradient-to-br from-blue-50/50 to-transparent dark:from-blue-900/20 dark:to-transparent overflow-hidden">
              <div className="absolute top-0 right-0 w-40 h-40 bg-blue-200/30 dark:bg-blue-500/10 rounded-full blur-2xl -mr-20 -mt-20 z-0"></div>
              <CardHeader className="relative z-10">
                <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900/40 inline-block mb-3">
                  <Code2 className="h-8 w-8 text-blue-600 dark:text-blue-400" />
                </div>
                <CardTitle className="text-xl text-blue-800 dark:text-blue-300">{t('home.freeCourses.cards.python.title')}</CardTitle>
                <CardDescription className="text-blue-700/80 dark:text-blue-400/80">
                  {t('home.freeCourses.cards.python.description')}
                </CardDescription>
              </CardHeader>
              <CardFooter className="relative z-10">
                <Button asChild variant="default" className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                  <Link href="/topics/python/">
                    {t('home.freeCourses.startLearning')} <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>

            <Card className="group hover:shadow-lg transition-all duration-300 border-purple-200 dark:border-purple-900/40 bg-gradient-to-br from-purple-50/50 to-transparent dark:from-purple-900/20 dark:to-transparent overflow-hidden">
              <div className="absolute top-0 right-0 w-40 h-40 bg-purple-200/30 dark:bg-purple-500/10 rounded-full blur-2xl -mr-20 -mt-20 z-0"></div>
              <CardHeader className="relative z-10">
                <div className="p-3 rounded-lg bg-purple-100 dark:bg-purple-900/40 inline-block mb-3">
                  <Brain className="h-8 w-8 text-purple-600 dark:text-purple-400" />
                </div>
                <CardTitle className="text-xl text-purple-800 dark:text-purple-300">{t('home.freeCourses.cards.intro.title')}</CardTitle>
                <CardDescription className="text-purple-700/80 dark:text-purple-400/80">
                  {t('home.freeCourses.cards.intro.description')}
                </CardDescription>
              </CardHeader>
              <CardFooter className="relative z-10">
                <Button asChild variant="default" className="w-full bg-purple-600 hover:bg-purple-700 text-white">
                  <Link href="/topics/ai-fundamentals">
                    {t('home.freeCourses.startLearning')} <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>

            <Card className="group hover:shadow-lg transition-all duration-300 border-green-200 dark:border-green-900/40 bg-gradient-to-br from-green-50/50 to-transparent dark:from-green-900/20 dark:to-transparent overflow-hidden">
              <div className="absolute top-0 right-0 w-40 h-40 bg-green-200/30 dark:bg-green-500/10 rounded-full blur-2xl -mr-20 -mt-20 z-0"></div>
              <CardHeader className="relative z-10">
                <div className="p-3 rounded-lg bg-green-100 dark:bg-green-900/40 inline-block mb-3">
                  <Brain className="h-8 w-8 text-green-600 dark:text-green-400" />
                </div>
                <CardTitle className="text-xl text-green-800 dark:text-green-300">{t('home.freeCourses.cards.neural.title')}</CardTitle>
                <CardDescription className="text-green-700/80 dark:text-green-400/80">
                  {t('home.freeCourses.cards.neural.description')}
                </CardDescription>
              </CardHeader>
              <CardFooter className="relative z-10">
                <Button asChild variant="default" className="w-full bg-green-600 hover:bg-green-700 text-white">
                  <Link href="/neural-networks/basics">
                    {t('home.freeCourses.startLearning')} <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>
          </div>

          <div className="text-center">
            <Button asChild size="lg" variant="outline" className="rounded-full">
              <Link href="/topics">
                {t('home.freeCourses.viewAll')} <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Topics section */}
      <section className="relative z-30 py-16 md:py-20 bg-muted/30" aria-labelledby="topics-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-12 md:mb-16">
            <h2 id="topics-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">
              {t('home.topics.title')}
            </h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
              {t('home.topics.subtitle')}
            </p>
          </div>
          
          <div className="flex flex-col gap-8">
            {/* Featured Topics - Python & AI */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
              {/* Python Topic - Featured */}
              <Card className="overflow-hidden transition-all duration-300 hover:shadow-xl hover:-translate-y-1.5 border-blue-200/50 dark:border-blue-800/50 bg-card flex flex-col">
                <div className="relative h-64 w-full">
                  <Image 
                    src="https://images.pexels.com/photos/1181671/pexels-photo-1181671.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                    alt={`Python Programming - ${t('common.imageAlt')}`}
                    fill
                    className="object-cover transition-transform duration-300 group-hover:scale-105"
                    loading="eager"
                    sizes="(max-width: 768px) 100vw, 50vw"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent" aria-hidden="true" />
                  <div className="absolute bottom-4 left-4 p-3 rounded-full bg-background/80 backdrop-blur-sm border border-border shadow-md">
                    <Code2 className="h-8 w-8 text-blue-500" />
                  </div>
                </div>
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-center mb-2">
                    <CardTitle className="text-2xl group-hover:text-primary transition-colors">Python Programlama</CardTitle>
                    <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300">
                      Temel
                    </span>
                  </div>
                  <CardDescription className="text-base">
                    Yapay zeka ve veri bilimi için temel Python programlama becerilerini öğrenin. Veri yapıları, nesne tabanlı programlama ve daha fazlası.
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex-grow">
                  <div className="flex flex-wrap gap-2 mt-2">
                    <span className="px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs rounded-md">Temel Sözdizimi</span>
                    <span className="px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs rounded-md">Veri Yapıları</span>
                    <span className="px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs rounded-md">OOP</span>
                    <span className="px-2 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs rounded-md">NumPy & Pandas</span>
                  </div>
                </CardContent>
                <CardFooter className="mt-auto pt-3">
                  <Button asChild variant="default" className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                    <Link href="/topics/python">
                      Python Derslerine Git
                      <ArrowRight className="h-4 w-4 ml-2" aria-hidden="true" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>

              {/* AI Topic - Featured */}
              <Card className="overflow-hidden transition-all duration-300 hover:shadow-xl hover:-translate-y-1.5 border-purple-200/50 dark:border-purple-800/50 bg-card flex flex-col">
                <div className="relative h-64 w-full">
                  <Image 
                    src="https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                    alt={`Yapay Zeka - ${t('common.imageAlt')}`}
                    fill
                    className="object-cover transition-transform duration-300 group-hover:scale-105"
                    loading="eager"
                    sizes="(max-width: 768px) 100vw, 50vw"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent" aria-hidden="true" />
                  <div className="absolute bottom-4 left-4 p-3 rounded-full bg-background/80 backdrop-blur-sm border border-border shadow-md">
                    <Brain className="h-8 w-8 text-purple-500" />
                  </div>
                </div>
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-center mb-2">
                    <CardTitle className="text-2xl group-hover:text-primary transition-colors">Yapay Zeka</CardTitle>
                    <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300">
                      Kapsamlı
                    </span>
                  </div>
                  <CardDescription className="text-base">
                    Yapay zeka teknolojilerinin temellerini ve uygulamalarını keşfedin. Makine öğrenmesi, derin öğrenme ve daha fazlası.
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex-grow">
                  <div className="flex flex-wrap gap-2 mt-2">
                    <span className="px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 text-xs rounded-md">Makine Öğrenmesi</span>
                    <span className="px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 text-xs rounded-md">Derin Öğrenme</span>
                    <span className="px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 text-xs rounded-md">NLP</span>
                    <span className="px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 text-xs rounded-md">Bilgisayarlı Görü</span>
                  </div>
                </CardContent>
                <CardFooter className="mt-auto pt-3">
                  <Button asChild variant="default" className="w-full bg-purple-600 hover:bg-purple-700 text-white">
                    <Link href="/topics/ai-fundamentals">
                      Yapay Zeka Derslerine Git
                      <ArrowRight className="h-4 w-4 ml-2" aria-hidden="true" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            </div>

            {/* Other Topics */}
            <h3 className="text-xl font-semibold text-center mt-8 mb-6">Diğer Konular</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
              {topics.filter(topic => 
                !topic.href.includes('/python') && 
                !topic.href.includes('/ai-fundamentals')
              ).map((topic, index) => (
                <Card key={index} className="overflow-hidden transition-all duration-300 hover:shadow-xl hover:-translate-y-1.5 border-border hover:border-primary/50 bg-card flex flex-col">
                  <div className="relative h-52 w-full">
                    <Image 
                      src={topic.imageUrl}
                      alt={`${topic.title} - ${t('common.imageAlt')}`}
                      fill
                      className="object-cover transition-transform duration-300 group-hover:scale-105"
                      loading={index < 3 ? "eager" : "lazy"}
                      sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent" aria-hidden="true" />
                    <div className="absolute bottom-4 left-4 p-3 rounded-full bg-background/80 backdrop-blur-sm border border-border shadow-md">
                      {topic.icon}
                    </div>
                  </div>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xl group-hover:text-primary transition-colors">{topic.title}</CardTitle>
                  </CardHeader>
                  <CardContent className="flex-grow">
                    <CardDescription>{topic.description}</CardDescription>
                  </CardContent>
                  <CardFooter className="mt-auto pt-3">
                    <Button asChild variant="ghost" className="gap-1.5 ml-auto text-primary hover:text-primary/80">
                      <Link href={topic.href}>
                        {t('common.exploreTopic')}
                        <ArrowRight className="h-4 w-4" aria-hidden="true" />
                      </Link>
                    </Button>
                  </CardFooter>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </section>
      
      {/* Latest Blog Posts Section */}
      <section className="relative z-30 py-16 md:py-20 bg-background" aria-labelledby="latest-blog-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-12 md:mb-16">
            <div className="inline-block p-3 mb-4 bg-primary/10 rounded-full border border-primary/20">
                <Rss className="h-10 w-10 text-primary" />
            </div>
            <h2 id="latest-blog-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">{t('home.blog.title')}</h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
              {t('home.blog.subtitle')}
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
            {latestBlogPosts.map((post, index) => (
              <Card key={index} className="overflow-hidden transition-all duration-300 hover:shadow-xl hover:-translate-y-1 group bg-card border-border hover:border-primary/50 flex flex-col">
                <Link href={post.href} className="block relative h-52 w-full group-hover:opacity-90 transition-opacity">
                  <Image 
                    src={post.imageUrl} 
                    alt={`${post.title} - ${t('common.imageAlt')}`}
                    fill 
                    className="object-cover transition-transform duration-300 group-hover:scale-105"
                    loading={index < 3 ? "eager" : "lazy"}
                    sizes="(max-width: 768px) 100vw, 33vw"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/40 to-transparent" aria-hidden="true"></div>
                </Link>
                <CardHeader className="pb-2">
                  <p className="text-xs text-muted-foreground mb-1">
                    <span className="font-semibold text-primary">{post.category}</span> &bull; {post.date}
                  </p>
                  <CardTitle className="text-lg leading-tight group-hover:text-primary transition-colors">
                    <Link href={post.href}>{post.title}</Link>
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-grow pt-0">
                  <p className="text-sm text-muted-foreground line-clamp-3">{post.snippet}</p>
                </CardContent>
                <CardFooter className="mt-auto pt-3">
                  <Button asChild variant="link" className="p-0 h-auto text-primary hover:text-primary/80">
                    <Link href={post.href}>
                      {t('common.readMore')}
                      <ArrowRight className="ml-1 h-4 w-4" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
          <div className="mt-12 text-center">
            <Button asChild size="lg" className="rounded-full shadow-lg hover:shadow-xl transition-shadow">
              <Link href="/blog">
                {t('home.blog.viewAll')}
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Why Kodleon section */}
      <section className="relative z-30 py-16 md:py-20 bg-muted/30" aria-labelledby="why-kodleon-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-12 md:mb-16">
            <div className="inline-block p-3 mb-4 bg-primary/10 rounded-full border border-primary/20">
              <Brain className="h-10 w-10 text-primary" />
            </div>
            <h2 id="why-kodleon-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">{t('home.whyKodleon.title')}</h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
              {t('home.whyKodleon.subtitle')}
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-10">
            {whyKodleonFeatures.map((feature, index) => (
              <div key={index} className="flex items-start gap-4">
                <div className="flex-shrink-0 p-3 bg-background rounded-full border shadow-md">
                  {feature.icon}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-foreground mb-1">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      {/* What you can achieve */}
      <section className="relative z-30 py-16 md:py-20 bg-background" aria-labelledby="achieve-heading">
        <div className="container max-w-6xl mx-auto px-4">
          <div className="text-center mb-12 md:mb-16">
            <h2 id="achieve-heading" className="text-3xl md:text-4xl font-bold tracking-tight mb-4 text-foreground">
              {t('home.achieve.title')}
            </h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto">
              {t('home.achieve.subtitle')}
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 md:gap-8">
            {whatYouCanAchieve.map((item, index) => (
              <div key={index} className="p-6 bg-card border border-border rounded-lg shadow-lg text-center hover:shadow-primary/20 transition-shadow">
                <div className="flex justify-center items-center mb-4 bg-primary/10 p-3 rounded-full w-16 h-16 mx-auto border border-primary/20">
                  {item.icon} 
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">{item.title}</h3>
                <p className="text-sm text-muted-foreground">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Join Us Section */}
      <section className="py-16 md:py-32 bg-gradient-to-br from-primary via-purple-600 to-secondary text-primary-foreground" aria-labelledby="join-us-heading">
        <div 
          className="absolute inset-0 opacity-20 dark:opacity-10"
          style={{
            backgroundImage: 'url("data:image/svg+xml,%3Csvg width="52" height="52" viewBox="0 0 52 52" xmlns="http://www.w3.org/2000/svg"%3E%3Cpath d="M0 26L26 0L52 26L26 52Z" fill="%23FFFFFF" fill-opacity="0.1"/%3E%3C/svg%3E")',
            backgroundSize: '52px 52px'
          }}
          aria-hidden="true"
        />
        <div className="container max-w-3xl mx-auto text-center relative z-10 px-4">
          <h2 id="join-us-heading" className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
            {t('home.joinUs.title')}
          </h2>
          <p className="text-xl md:text-2xl text-primary-foreground/80 mb-10">
            {t('home.joinUs.subtitle')}
          </p>
          <Button asChild size="lg" variant="outline" className="bg-primary-foreground text-primary hover:bg-primary-foreground/90 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 text-lg px-10 py-7">
            <Link href="/topics">
              {t('home.joinUs.cta')}
              <Zap className="ml-2 h-5 w-5" />
            </Link>
          </Button>
        </div>
      </section>
    </div>
  );
} 