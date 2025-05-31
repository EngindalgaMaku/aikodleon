'use client';

import { usePathname, useSearchParams } from 'next/navigation';
import tr from '../locales/tr.json';
import en from '../locales/en.json';

const translations = {
  tr,
  en,
};

export type Locale = 'tr' | 'en';

export function useTranslation() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const locale = (searchParams.get('locale') as Locale) || 'tr';
  
  const t = (key: string) => {
    const keys = key.split('.');
    let value: any = translations[locale];
    
    for (const k of keys) {
      if (value && typeof value === 'object') {
        value = value[k];
      } else {
        return key;
      }
    }
    
    return value || key;
  };

  const changeLocale = (newLocale: Locale) => {
    const params = new URLSearchParams(searchParams.toString());
    if (newLocale === 'tr') {
      params.delete('locale');
    } else {
      params.set('locale', newLocale);
    }
    
    const queryString = params.toString();
    window.location.href = queryString ? `${pathname}?${queryString}` : pathname;
  };

  return {
    t,
    locale,
    changeLocale,
  };
} 