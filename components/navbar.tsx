"use client";

import { useState } from "react";
import Link from "next/link";
import { Brain, Menu, X } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { Button } from "./ui/button";
import { useTranslation } from "@/lib/i18n";

export default function Navbar() {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);

  const routes = [
    { name: t('navigation.home'), href: "/" },
    { name: t('navigation.topics'), href: "/topics" },
    { name: t('navigation.blog'), href: "/blog" },
    { name: t('navigation.about'), href: "/about" },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container max-w-6xl mx-auto flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2" aria-label="Kodleon ana sayfaya git">
          <Brain className="h-6 w-6 text-primary" aria-hidden="true" />
          <span className="font-bold text-xl hidden sm:inline-block">Kodleon</span>
        </Link>
        
        <nav className="hidden md:flex items-center gap-6" aria-label="Ana navigasyon">
          {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              className="text-sm font-medium transition-colors hover:text-primary"
            >
              {route.name}
            </Link>
          ))}
          <ThemeToggle />
          <Button variant="default" asChild>
            <Link href="/contact">{t('navigation.contact')}</Link>
          </Button>
        </nav>
        
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          aria-controls="mobile-menu"
          aria-label="Menüyü aç/kapat"
        >
          {isOpen ? <X className="h-5 w-5" aria-hidden="true" /> : <Menu className="h-5 w-5" aria-hidden="true" />}
          <span className="sr-only">Menüyü aç/kapat</span>
        </Button>

        {isOpen && (
          <div className="absolute top-full left-0 right-0 bg-background border-b md:hidden">
            <nav className="container max-w-6xl mx-auto py-4 flex flex-col gap-4">
              {routes.map((route) => (
                <Link
                  key={route.href}
                  href={route.href}
                  className="text-sm font-medium transition-colors hover:text-primary px-4 py-2"
                  onClick={() => setIsOpen(false)}
                >
                  {route.name}
                </Link>
              ))}
              <Button variant="default" asChild className="mx-4">
                <Link href="/contact">{t('navigation.contact')}</Link>
              </Button>
              <div className="px-4">
                <ThemeToggle />
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}