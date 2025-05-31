'use client';

import { useTranslation, type Locale } from '@/lib/i18n';
import { Globe } from 'lucide-react';
import { TR, GB } from 'country-flag-icons/react/3x2';
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

export function LanguageSwitcher() {
  const { locale, changeLocale } = useTranslation();

  const handleLocaleChange = (newLocale: string) => {
    if (newLocale && (newLocale === 'tr' || newLocale === 'en')) {
      changeLocale(newLocale as Locale);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <Globe className="h-4 w-4" />
      <ToggleGroup
        type="single"
        value={locale}
        onValueChange={handleLocaleChange}
        aria-label="Language selection"
        size="sm"
      >
        <ToggleGroupItem value="tr" aria-label="Türkçe">
          <TR title="Türkçe" className="h-4 w-4 rounded-sm" />
        </ToggleGroupItem>
        <ToggleGroupItem value="en" aria-label="English">
          <GB title="English" className="h-4 w-4 rounded-sm" />
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  );
} 