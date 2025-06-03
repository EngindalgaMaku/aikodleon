"use client";

import { useState } from "react";
import Link from "next/link";
import { Brain, Menu, X, Rocket } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { Button } from "./ui/button";
import { useTranslation } from "@/lib/i18n";
import { SiPython } from "react-icons/si";

export default function Navbar() {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);

  const routes = [
    { name: t('navigation.home'), href: "/" },
    { name: t('navigation.topics'), href: "/topics" },
    { name: t('navigation.blog'), href: "/blog" },
  ];

  const practicalExamplesLink = "/pratik-ornekler";
  const practicalExamplesText = "Kod Örnekleri";

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container max-w-6xl mx-auto flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2" aria-label="Kodleon ana sayfaya git">
          <Brain className="h-6 w-6 text-primary" aria-hidden="true" />
          <span className="font-bold text-xl hidden sm:inline-block">Kodleon</span>
        </Link>
        
        <nav className="hidden md:flex items-center gap-4" aria-label="Ana navigasyon">
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
          <Link
            href="/topics/python"
            className="flex items-center gap-1.5 text-sm font-bold text-blue-600 hover:text-white bg-blue-100 hover:bg-blue-600 px-3 py-1.5 rounded transition-colors shadow-sm"
            style={{ fontFamily: 'monospace' }}
          >
            <SiPython className="h-5 w-5 text-blue-500" />
            Python Dersleri
          </Link>
          <Button variant="default" asChild size="sm">
            <Link href={practicalExamplesLink} className="flex items-center gap-1.5">
              <Rocket className="h-4 w-4" />
              {practicalExamplesText}
            </Link>
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
                className="flex items-center gap-2 text-base font-bold transition-colors px-4 py-2 rounded-md bg-blue-100 text-blue-600 hover:bg-blue-600 hover:text-white"
                style={{ fontFamily: 'monospace' }}
                onClick={() => setIsOpen(false)}
              >
                <SiPython className="h-5 w-5 text-blue-500" />
                Python Dersleri
              </Link>
              <Link 
                href={practicalExamplesLink}
                className="flex items-center gap-2 text-base font-medium transition-colors hover:text-primary px-4 py-2 rounded-md hover:bg-muted bg-primary/10 text-primary"
                onClick={() => setIsOpen(false)}
              >
                <Rocket className="h-5 w-5" />
                {practicalExamplesText}
              </Link>
              <div className="pt-2 flex justify-start pl-4">
                <ThemeToggle />
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}