'use client';

import { useTranslation, type Locale } from '@/lib/i18n';
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
    <ToggleGroup
      type="single"
      value={locale}
      onValueChange={handleLocaleChange}
      aria-label="Language selection"
      size="sm"
      className="h-8"
    >
      <ToggleGroupItem value="tr" aria-label="Türkçe" className="px-2">
        <TR title="Türkçe" className="h-3.5 w-3.5 rounded-sm" />
      </ToggleGroupItem>
      <ToggleGroupItem value="en" aria-label="English" className="px-2">
        <GB title="English" className="h-3.5 w-3.5 rounded-sm" />
      </ToggleGroupItem>
    </ToggleGroup>
  );
} 