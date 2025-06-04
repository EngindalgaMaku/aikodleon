import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Frontend Entegrasyonu | Python Web Geliştirme | Kodleon',
  description: 'Modern frontend teknolojilerinin Python web uygulamalarıyla entegrasyonu. React, TypeScript ve API entegrasyonu konuları.',
};

const content = `
# Frontend Entegrasyonu

Modern web uygulamalarında frontend geliştirme ve Python backend servisleriyle entegrasyon süreçlerini öğreneceğiz.

## React ve TypeScript Kurulumu

\`\`\`bash
# Next.js projesi oluşturma
npx create-next-app@latest frontend --typescript --tailwind --eslint

# Gerekli bağımlılıkları kurma
cd frontend
npm install @tanstack/react-query axios zod @hookform/resolvers/zod react-hook-form
\`\`\`

## API İstemci Konfigürasyonu

\`\`\`typescript
// lib/api.ts
import axios from 'axios';

export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// İstek interceptor'ı
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = \`Bearer \${token}\`;
  }
  return config;
});

// Yanıt interceptor'ı
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Token yenileme veya çıkış yapma
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
\`\`\`

## Veri Doğrulama ve Form Yönetimi

\`\`\`typescript
// schemas/post.ts
import { z } from 'zod';

export const postSchema = z.object({
  title: z.string().min(3, 'Başlık en az 3 karakter olmalıdır'),
  content: z.string().min(10, 'İçerik en az 10 karakter olmalıdır'),
  category: z.string().uuid('Geçerli bir kategori seçin'),
});

export type PostFormData = z.infer<typeof postSchema>;

// components/PostForm.tsx
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { postSchema, type PostFormData } from '@/schemas/post';
import { api } from '@/lib/api';

export function PostForm() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<PostFormData>({
    resolver: zodResolver(postSchema),
  });

  const onSubmit = async (data: PostFormData) => {
    try {
      await api.post('/posts', data);
      // Başarılı işlem
    } catch (error) {
      // Hata yönetimi
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <label>Başlık</label>
        <input {...register('title')} />
        {errors.title && <span>{errors.title.message}</span>}
      </div>
      
      <div>
        <label>İçerik</label>
        <textarea {...register('content')} />
        {errors.content && <span>{errors.content.message}</span>}
      </div>
      
      <div>
        <label>Kategori</label>
        <select {...register('category')}>
          {/* Kategori seçenekleri */}
        </select>
        {errors.category && <span>{errors.category.message}</span>}
      </div>
      
      <button type="submit">Gönder</button>
    </form>
  );
}
\`\`\`

## Veri Yönetimi ve Caching

\`\`\`typescript
// hooks/usePosts.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { PostFormData } from '@/schemas/post';

export function usePosts() {
  const queryClient = useQueryClient();

  const posts = useQuery({
    queryKey: ['posts'],
    queryFn: () => api.get('/posts').then((res) => res.data),
  });

  const createPost = useMutation({
    mutationFn: (data: PostFormData) => api.post('/posts', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['posts'] });
    },
  });

  const updatePost = useMutation({
    mutationFn: ({ id, data }: { id: string; data: PostFormData }) =>
      api.put(\`/posts/\${id}\`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['posts'] });
    },
  });

  const deletePost = useMutation({
    mutationFn: (id: string) => api.delete(\`/posts/\${id}\`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['posts'] });
    },
  });

  return {
    posts,
    createPost,
    updatePost,
    deletePost,
  };
}
\`\`\`

## Kimlik Doğrulama ve Yetkilendirme

\`\`\`typescript
// hooks/useAuth.ts
import { create } from 'zustand';
import { api } from '@/lib/api';

interface AuthState {
  user: any | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

export const useAuth = create<AuthState>((set) => ({
  user: null,
  token: localStorage.getItem('token'),
  
  login: async (email: string, password: string) => {
    const response = await api.post('/auth/login', { email, password });
    const { token, user } = response.data;
    
    localStorage.setItem('token', token);
    set({ token, user });
  },
  
  logout: () => {
    localStorage.removeItem('token');
    set({ token: null, user: null });
  },
}));

// components/ProtectedRoute.tsx
import { useAuth } from '@/hooks/useAuth';
import { useRouter } from 'next/router';

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { token } = useAuth();
  const router = useRouter();

  if (!token) {
    router.push('/login');
    return null;
  }

  return <>{children}</>;
}
\`\`\`

## Alıştırmalar

1. **API Entegrasyonu**
   - CRUD operasyonları için API istemcisi oluşturun
   - Error handling ve loading state yönetimi yapın
   - API response caching implementasyonu yapın
   - Interceptor'lar ile request/response yönetimi yapın

2. **Form Yönetimi**
   - Kompleks form validasyonu oluşturun
   - Form state yönetimi implementasyonu yapın
   - File upload özelliği ekleyin
   - Dynamic form fields oluşturun

3. **State Management**
   - Global state yönetimi kurun
   - Server state ile client state senkronizasyonu yapın
   - Optimistic updates implementasyonu yapın
   - Real-time updates ekleyin

## Sonraki Adımlar

1. [Mikroservis Mimarisi](/topics/python/web-gelistirme/microservices)
2. [Cloud Native Development](/topics/python/web-gelistirme/cloud-native)
3. [Progressive Web Apps](/topics/python/web-gelistirme/pwa)

## Faydalı Kaynaklar

- [React Dokümantasyonu](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [TanStack Query Dokümantasyonu](https://tanstack.com/query/latest)
- [Zod Dokümantasyonu](https://zod.dev/)
`;

const learningPath = [
  {
    title: '1. Modern Frontend',
    description: 'Modern frontend teknolojilerini öğrenin.',
    topics: [
      'React temelleri',
      'TypeScript ile tip güvenliği',
      'Next.js framework',
      'Tailwind CSS',
      'Component mimarisi',
    ],
    icon: '⚛️',
    href: '/topics/python/web-gelistirme/frontend/react'
  },
  {
    title: '2. API Entegrasyonu',
    description: 'Backend servisleriyle entegrasyonu öğrenin.',
    topics: [
      'REST API client',
      'Authentication/Authorization',
      'Error handling',
      'Data fetching',
      'Caching stratejileri',
    ],
    icon: '🔌',
    href: '/topics/python/web-gelistirme/frontend/api'
  },
  {
    title: '3. State Yönetimi',
    description: 'Uygulama state yönetimini öğrenin.',
    topics: [
      'React Query',
      'Zustand',
      'Form state',
      'Server state',
      'Performance optimizasyonu',
    ],
    icon: '📊',
    href: '/topics/python/web-gelistirme/frontend/state'
  },
  {
    title: '4. Testing ve Deployment',
    description: 'Frontend test ve deployment süreçlerini öğrenin.',
    topics: [
      'Unit testing',
      'Integration testing',
      'CI/CD pipeline',
      'Static hosting',
      'Performance monitoring',
    ],
    icon: '🚀',
    href: '/topics/python/web-gelistirme/frontend/testing'
  }
];

export default function FrontendPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/web-gelistirme" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Web Geliştirme
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert">
          <MarkdownContent content={content} />
        </div>

        <h2 className="text-2xl font-bold mb-6">Öğrenme Yolu</h2>
        
        <div className="grid gap-6 md:grid-cols-2">
          {learningPath.map((topic, index) => (
            <Card key={index} className="p-6 hover:bg-accent transition-colors cursor-pointer">
              <Link href={topic.href}>
                <div className="flex items-start space-x-4">
                  <div className="text-4xl">{topic.icon}</div>
                  <div className="space-y-2">
                    <h3 className="font-bold">{topic.title}</h3>
                    <p className="text-sm text-muted-foreground">{topic.description}</p>
                    <ul className="text-sm space-y-1 list-disc list-inside text-muted-foreground">
                      {topic.topics.map((t, i) => (
                        <li key={i}>{t}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Link>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 