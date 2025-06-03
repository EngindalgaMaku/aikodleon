"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Brain, Menu, X } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { Button } from "./ui/button";
import { useTranslation } from "@/lib/i18n";
import { SiPython } from "react-icons/si";
import { LanguageSwitcher } from "./LanguageSwitcher";

export default function Navbar() {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();
  const isHomePage = pathname === "/";

  const routes = [
    ...(isHomePage ? [] : [{ name: t('navigation.home'), href: "/" }]),
    { name: t('navigation.topics'), href: "/topics" },
    { name: t('navigation.blog'), href: "/blog" },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container max-w-6xl mx-auto flex h-16 items-center">
        {/* Logo - Sol */}
        <div className="flex-none">
          <Link href="/" className="flex items-center gap-2" aria-label="Kodleon ana sayfaya git">
            <Brain className="h-6 w-6 text-primary" aria-hidden="true" />
            <span className="font-bold text-xl hidden sm:inline-block">Kodleon</span>
          </Link>
        </div>
        
        {/* Ana Menü - Orta */}
        <nav className="hidden md:flex flex-1 items-center justify-center" aria-label="Ana navigasyon">
          {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              className="text-sm font-medium transition-colors hover:text-primary px-4"
            >
              {route.name}
            </Link>
          ))}
          
          <Button asChild variant="ghost" size="sm" className="bg-blue-100 hover:bg-blue-200 text-blue-600 mx-2">
            <Link href="/topics/python" className="flex items-center gap-1.5">
              <SiPython className="h-4 w-4" />
              Python Dersleri
            </Link>
          </Button>
          <Button asChild variant="default" size="sm" className="mx-2">
            <Link href="/kod-ornekleri" className="flex items-center gap-1.5">
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
                className="flex items-center gap-2 text-base font-medium px-4 py-2 rounded-md bg-blue-100 text-blue-600 hover:bg-blue-200"
                onClick={() => setIsOpen(false)}
              >
                <SiPython className="h-5 w-5" />
                Python Dersleri
              </Link>
              <Link 
                href="/kod-ornekleri"
                className="flex items-center gap-2 text-base font-medium px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
                onClick={() => setIsOpen(false)}
              >
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