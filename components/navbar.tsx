"use client";

import { useState } from "react";
import Link from "next/link";
import { Brain, Menu, X } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

const routes = [
  { name: "Ana Sayfa", href: "/" },
  { name: "Konular", href: "/topics" },
  { name: "Hakkında", href: "/about" },
];

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

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
          <div className="absolute top-16 left-0 right-0 border-b bg-background md:hidden" id="mobile-menu">
            <nav className="container max-w-6xl mx-auto flex flex-col py-4 gap-2" aria-label="Mobil navigasyon">
              {routes.map((route) => (
                <Link
                  key={route.href}
                  href={route.href}
                  className="px-4 py-2 text-sm font-medium transition-colors hover:text-primary"
                  onClick={() => setIsOpen(false)}
                >
                  {route.name}
                </Link>
              ))}
              <div className="px-4 pt-2">
                <ThemeToggle />
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}