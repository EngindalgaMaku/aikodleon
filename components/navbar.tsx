"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { Brain, Menu, X, Home, Code, Sparkles } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { Button } from "./ui/button";
import { useTranslation } from "@/lib/i18n";
import { SiPython } from "react-icons/si";
import { LanguageSwitcher } from "./LanguageSwitcher";

export default function Navbar() {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const pathname = usePathname();
  const isHomePage = pathname === "/";

  // Scroll efekti
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const routes = [
    { name: t('navigation.topics'), href: "/topics" },
    { name: t('navigation.blog'), href: "/blog" },
  ];

  return (
    <header className={`sticky top-0 z-50 w-full border-b backdrop-blur supports-[backdrop-filter]:bg-background/60 transition-all duration-200 ${
      scrolled ? "bg-background/95 shadow-sm" : "bg-background/80"
    }`}>
      <div className="container max-w-6xl mx-auto flex h-16 items-center">
        {/* Logo - Sol */}
        <div className="flex-none">
          <Link href="/" className="flex items-center gap-2" aria-label="Kodleon ana sayfaya git">
            <div className="relative h-9 w-9 overflow-hidden">
              {/* SVG logo kullanımı */}
              <svg width="36" height="36" viewBox="0 0 512 512" fill="none" xmlns="http://www.w3.org/2000/svg" className="rounded-md">
                <path d="M256 50C142.5 50 50 142.5 50 256C50 369.5 142.5 462 256 462C369.5 462 462 369.5 462 256C462 142.5 369.5 50 256 50Z" fill="#5046E5" opacity="0.1"/>
                <path d="M130 180L210 140M210 140L290 180M290 180L370 140M210 140L210 230M290 180L290 230M210 230L290 230M210 230L170 300M290 230L330 300M170 300L250 350M330 300L250 350M250 350L250 400" stroke="#5046E5" strokeWidth="6" strokeLinecap="round"/>
                <circle cx="130" cy="180" r="12" fill="#5046E5"/>
                <circle cx="210" cy="140" r="12" fill="#5046E5"/>
                <circle cx="290" cy="180" r="12" fill="#5046E5"/>
                <circle cx="370" cy="140" r="12" fill="#5046E5"/>
                <circle cx="210" cy="230" r="12" fill="#5046E5"/>
                <circle cx="290" cy="230" r="12" fill="#5046E5"/>
                <circle cx="170" cy="300" r="12" fill="#5046E5"/>
                <circle cx="330" cy="300" r="12" fill="#5046E5"/>
                <circle cx="250" cy="350" r="12" fill="#5046E5"/>
                <circle cx="250" cy="400" r="12" fill="#5046E5"/>
                <circle cx="210" cy="140" r="24" fill="#5046E5" opacity="0.3"/>
                <circle cx="290" cy="180" r="24" fill="#5046E5" opacity="0.3"/>
                <circle cx="250" cy="350" r="24" fill="#5046E5" opacity="0.3"/>
              </svg>
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-xl hidden sm:inline-block bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Kodleon</span>
              <span className="text-xs text-muted-foreground hidden sm:inline-block">AI Eğitim Platformu</span>
            </div>
          </Link>
        </div>
        
        {/* Ana Menü - Orta */}
        <nav className="hidden md:flex flex-1 items-center justify-center" aria-label="Ana navigasyon">
          {!isHomePage && (
            <Link
              href="/"
              className="text-sm font-medium transition-colors hover:text-primary px-4 flex items-center"
              aria-label={t('navigation.home')}
            >
              <Home className="h-4 w-4" />
            </Link>
          )}
          
          {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              className="text-sm font-medium transition-colors hover:text-primary px-4"
            >
              {route.name}
            </Link>
          ))}
          
          <Button asChild variant="ghost" size="sm" className="bg-gradient-to-r from-blue-50 to-indigo-50 hover:from-blue-100 hover:to-indigo-100 text-indigo-600 mx-2 border border-indigo-100">
            <Link href="/topics/python" className="flex items-center gap-1.5">
              <SiPython className="h-4 w-4" />
              Python Dersleri
            </Link>
          </Button>
          <Button asChild variant="default" size="sm" className="mx-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700">
            <Link href="/kod-ornekleri" className="flex items-center gap-1.5">
              <Code className="h-4 w-4 mr-1" />
              Kod Örnekleri
            </Link>
          </Button>
        </nav>

        {/* Dil ve Tema - Sağ */}
        <div className="hidden md:flex flex-none items-center gap-2">
          <LanguageSwitcher />
          <ThemeToggle />
        </div>
        
        {/* Mobil Menü Butonu */}
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden ml-auto"
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          aria-controls="mobile-menu"
          aria-label="Menüyü aç/kapat"
        >
          {isOpen ? <X className="h-5 w-5" aria-hidden="true" /> : <Menu className="h-5 w-5" aria-hidden="true" />}
          <span className="sr-only">Menüyü aç/kapat</span>
        </Button>

        {/* Mobil Menü */}
        {isOpen && (
          <div className="absolute top-full left-0 right-0 bg-background border-b md:hidden shadow-lg">
            <nav className="container max-w-6xl mx-auto py-4 flex flex-col gap-2">
              {!isHomePage && (
                <Link
                  href="/"
                  className="flex items-center gap-2 text-base font-medium px-4 py-2 rounded-md hover:bg-muted hover:text-primary"
                  onClick={() => setIsOpen(false)}
                  aria-label={t('navigation.home')}
                >
                  <Home className="h-5 w-5" />
                  {t('navigation.home')}
                </Link>
              )}
              
              {routes.map((route) => (
                <Link
                  key={route.href}
                  href={route.href}
                  className="text-base font-medium transition-colors hover:text-primary px-4 py-2 rounded-md hover:bg-muted"
                  onClick={() => setIsOpen(false)}
                >
                  {route.name}
                </Link>
              ))}
              <Link
                href="/topics/python"
                className="flex items-center gap-2 text-base font-medium px-4 py-2 rounded-md bg-gradient-to-r from-blue-50 to-indigo-50 text-indigo-600 border border-indigo-100"
                onClick={() => setIsOpen(false)}
              >
                <SiPython className="h-5 w-5" />
                Python Dersleri
              </Link>
              <Link 
                href="/kod-ornekleri"
                className="flex items-center gap-2 text-base font-medium px-4 py-2 rounded-md bg-gradient-to-r from-indigo-600 to-purple-600 text-white"
                onClick={() => setIsOpen(false)}
              >
                <Code className="h-5 w-5" />
                Kod Örnekleri
              </Link>
              <div className="pt-2 px-4 flex items-center gap-4">
                <LanguageSwitcher />
                <ThemeToggle />
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}