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
      title: t('home.blog.posts.embodiedAi.title'), // Assuming you'll add these specific keys
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
    },
    {
      title: t('home.blog.posts.aiVideoGeneration.title'),
      snippet: t('home.blog.posts.aiVideoGeneration.snippet'),
      imageUrl: "https://images.pexels.com/photos/9026285/pexels-photo-9026285.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
      href: "/blog/ai-video-uretimi-veo-flow",
      date: new Date().toLocaleDateString(locale, { year: 'numeric', month: 'long', day: 'numeric' }),
      category: t('home.blog.categories.generativeAi')
    }
  ];

  return (
    <div className="flex flex-col bg-background text-foreground">
      {/* Language Switcher */}
      <div className="absolute top-4 right-4 z-50">
        <LanguageSwitcher />
      </div>

      {/* Hero section */}
      <section className="relative py-20 md:py-32" aria-labelledby="hero-heading">
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
          <div className="max-w-3xl mx-auto text-center mb-12 md:mb-16">
            <h1 
              id="hero-heading" 
              className="text-4xl md:text-6xl font-bold tracking-tight mb-6 
                         bg-clip-text text-transparent bg-gradient-to-r from-primary via-pink-500 to-orange-500 
                         animate-gradient-xy"
            >
              {t('home.hero.title')}
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-10">
              {t('home.hero.subtitle')}
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild size="lg" className="rounded-full shadow-lg hover:shadow-xl transition-shadow">
                <Link href="/topics">
                  {t('home.hero.cta')}
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="rounded-full border-border hover:border-primary/70 transition-colors">
                <Link href="/blog">
                  <Rss className="mr-2 h-5 w-5" />
                  {t('navigation.blog')}
                </Link>
              </Button>
            </div>
          </div>

          <FeaturedBlogCarousel posts={latestBlogPosts} />
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
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
            {topics.map((topic, index) => (
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