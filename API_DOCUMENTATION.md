# Kodleon Platform - Comprehensive API Documentation

## Table of Contents
1. [Overview](#overview)
2. [UI Components](#ui-components)
3. [Layout Components](#layout-components)
4. [Content Components](#content-components)
5. [Utility Libraries](#utility-libraries)
6. [Custom Hooks](#custom-hooks)
7. [Page Components](#page-components)
8. [Types and Interfaces](#types-and-interfaces)
9. [Usage Examples](#usage-examples)

## Overview

Kodleon is a modern educational platform built with Next.js 13+, TypeScript, and Tailwind CSS. The platform focuses on AI, machine learning, and programming education in Turkish. It features a comprehensive design system, internationalization support, and advanced content management capabilities.

### Tech Stack
- **Framework**: Next.js 13+ (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS + shadcn/ui
- **UI Components**: Radix UI primitives
- **Content**: MDX + Markdown processing
- **Internationalization**: Custom i18n implementation
- **State Management**: React Hook Form + Custom hooks

## UI Components

### Button Component
**File**: `components/ui/button.tsx`

A versatile button component with multiple variants and sizes.

#### Props
```typescript
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}
```

#### Variants
- `default`: Primary button style
- `destructive`: For destructive actions
- `outline`: Outlined button
- `secondary`: Secondary button style
- `ghost`: Transparent button
- `link`: Link-style button

#### Sizes
- `default`: Standard size (h-10 px-4 py-2)
- `sm`: Small size (h-9 px-3)
- `lg`: Large size (h-11 px-8)
- `icon`: Icon-only size (h-10 w-10)

#### Usage
```typescript
import { Button } from '@/components/ui/button';

// Basic usage
<Button>Click me</Button>

// With variants
<Button variant="outline" size="lg">Large Outline Button</Button>

// As a child component
<Button asChild>
  <Link href="/page">Link Button</Link>
</Button>
```

### Card Components
**File**: `components/ui/card.tsx`

A set of components for creating card layouts.

#### Components
- `Card`: Main container
- `CardHeader`: Header section
- `CardTitle`: Title component
- `CardDescription`: Description text
- `CardContent`: Main content area
- `CardFooter`: Footer section

#### Usage
```typescript
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/components/ui/card';

<Card>
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
    <CardDescription>Card description goes here</CardDescription>
  </CardHeader>
  <CardContent>
    <p>Main content</p>
  </CardContent>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>
```

### Form Components
**File**: `components/ui/form.tsx`

Comprehensive form components built with React Hook Form integration.

#### Components
- `Form`: Form provider wrapper
- `FormField`: Field wrapper with validation
- `FormItem`: Form item container
- `FormLabel`: Form label with error states
- `FormControl`: Form control wrapper
- `FormDescription`: Help text
- `FormMessage`: Error message display

#### Usage
```typescript
import { useForm } from 'react-hook-form';
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';

const form = useForm();

<Form {...form}>
  <form onSubmit={form.handleSubmit(onSubmit)}>
    <FormField
      control={form.control}
      name="email"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Email</FormLabel>
          <FormControl>
            <Input placeholder="email@example.com" {...field} />
          </FormControl>
          <FormMessage />
        </FormItem>
      )}
    />
  </form>
</Form>
```

### Dialog Components
**File**: `components/ui/dialog.tsx`

Modal dialog components built on Radix UI.

#### Components
- `Dialog`: Root component
- `DialogTrigger`: Trigger button
- `DialogContent`: Modal content
- `DialogHeader`: Header section
- `DialogTitle`: Modal title
- `DialogDescription`: Modal description
- `DialogFooter`: Footer section
- `DialogClose`: Close button

#### Usage
```typescript
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';

<Dialog>
  <DialogTrigger asChild>
    <Button>Open Dialog</Button>
  </DialogTrigger>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Dialog Title</DialogTitle>
      <DialogDescription>Dialog description</DialogDescription>
    </DialogHeader>
    <div>Dialog content</div>
  </DialogContent>
</Dialog>
```

### Input Components
**File**: `components/ui/input.tsx`

Standard input component with consistent styling.

#### Usage
```typescript
import { Input } from '@/components/ui/input';

<Input type="email" placeholder="Enter your email" />
```

### Additional UI Components

#### Available Components
- `Accordion` - Collapsible content sections
- `Alert` - Status messages and notifications
- `Avatar` - User profile images
- `Badge` - Status indicators and labels
- `Breadcrumb` - Navigation breadcrumbs
- `Calendar` - Date picker component
- `Carousel` - Image/content carousel
- `Chart` - Data visualization components
- `Checkbox` - Checkbox input
- `Command` - Command palette/search
- `ContextMenu` - Right-click context menus
- `DropdownMenu` - Dropdown navigation menus
- `HoverCard` - Hover-triggered content
- `Label` - Form labels
- `Menubar` - Application menu bar
- `NavigationMenu` - Main navigation component
- `Pagination` - Page navigation
- `Popover` - Floating content containers
- `Progress` - Progress indicators
- `RadioGroup` - Radio button groups
- `ScrollArea` - Custom scrollable areas
- `Select` - Dropdown select component
- `Separator` - Visual dividers
- `Sheet` - Slide-out panels
- `Skeleton` - Loading placeholders
- `Slider` - Range input sliders
- `Switch` - Toggle switches
- `Table` - Data tables
- `Tabs` - Tabbed content
- `Textarea` - Multi-line text input
- `Toast` - Notification messages
- `Toggle` - Toggle buttons
- `ToggleGroup` - Toggle button groups
- `Tooltip` - Hover tooltips

## Layout Components

### Navbar Component
**File**: `components/navbar.tsx`

Main navigation component with responsive design and theme support.

#### Features
- Responsive mobile/desktop layouts
- Theme toggle integration
- Language switcher
- Scroll-based styling changes
- Accessibility support

#### Usage
```typescript
import Navbar from '@/components/navbar';

<Navbar />
```

### Footer Component
**File**: `components/footer.tsx`

Site footer with links and information.

#### Usage
```typescript
import Footer from '@/components/footer';

<Footer />
```

### Theme Provider
**File**: `components/theme-provider.tsx`

Theme management wrapper component.

#### Props
```typescript
interface ThemeProviderProps {
  children: React.ReactNode;
  attribute?: string;
  defaultTheme?: string;
  enableSystem?: boolean;
  disableTransitionOnChange?: boolean;
}
```

#### Usage
```typescript
import { ThemeProvider } from '@/components/theme-provider';

<ThemeProvider
  attribute="class"
  defaultTheme="system"
  enableSystem
  disableTransitionOnChange
>
  {children}
</ThemeProvider>
```

### Theme Toggle
**File**: `components/theme-toggle.tsx`

Theme switcher button component.

#### Usage
```typescript
import { ThemeToggle } from '@/components/theme-toggle';

<ThemeToggle />
```

## Content Components

### MarkdownContent Component
**File**: `components/MarkdownContent.tsx`

Advanced markdown renderer with syntax highlighting and math support.

#### Props
```typescript
interface MarkdownContentProps {
  content: string;
  className?: string;
}
```

#### Features
- Syntax highlighting with Prism
- Math equation support (KaTeX)
- GitHub Flavored Markdown (GFM)
- Custom styled components
- Responsive images
- Dark/light theme support

#### Usage
```typescript
import MarkdownContent from '@/components/MarkdownContent';

<MarkdownContent 
  content={markdownString}
  className="prose prose-lg"
/>
```

### LanguageSwitcher Component
**File**: `components/LanguageSwitcher.tsx`

Language selection component with flag icons.

#### Features
- Turkish/English language switching
- Country flag icons
- Toggle group interface
- Accessibility support

#### Usage
```typescript
import { LanguageSwitcher } from '@/components/LanguageSwitcher';

<LanguageSwitcher />
```

### TopicCard Component
**File**: `components/topic-card.tsx`

Card component for displaying topic information.

#### Props
```typescript
interface TopicCardProps {
  title: string;
  description: string;
  href: string;
}
```

#### Usage
```typescript
import { TopicCard } from '@/components/topic-card';

<TopicCard 
  title="Python Basics"
  description="Learn Python fundamentals"
  href="/topics/python-basics"
/>
```

### HomePageClientContent Component
**File**: `components/HomePageClientContent.tsx`

Main homepage content component with client-side features.

#### Usage
```typescript
import HomePageClientContent from '@/components/HomePageClientContent';

<HomePageClientContent />
```

### FeaturedBlogCarousel Component
**File**: `components/FeaturedBlogCarousel.tsx`

Carousel component for showcasing featured blog posts.

#### Usage
```typescript
import FeaturedBlogCarousel from '@/components/FeaturedBlogCarousel';

<FeaturedBlogCarousel />
```

### TopicSubtopicsList Component
**File**: `components/TopicSubtopicsList.tsx`

List component for displaying topic subtopics.

#### Props
```typescript
interface TopicSubtopicsListProps {
  subtopics: Array<{
    title: string;
    description: string;
    href: string;
  }>;
  isMetasearchTopic?: boolean;
  subtopicIcons?: Record<string, React.ComponentType>;
}
```

#### Usage
```typescript
import TopicSubtopicsList from '@/components/TopicSubtopicsList';

<TopicSubtopicsList 
  subtopics={subtopicsData}
  isMetasearchTopic={false}
  subtopicIcons={iconMap}
/>
```

### PaginatedSubtopicsList Component
**File**: `components/PaginatedSubtopicsList.tsx`

Paginated list component for large sets of subtopics.

#### Usage
```typescript
import PaginatedSubtopicsList from '@/components/PaginatedSubtopicsList';

<PaginatedSubtopicsList />
```

### TranslatedContent Component
**File**: `components/TranslatedContent.tsx`

Component for displaying translated content based on current locale.

#### Usage
```typescript
import { TranslatedContent } from '@/components/TranslatedContent';

<TranslatedContent />
```

## Utility Libraries

### Utils Library
**File**: `lib/utils.ts`

Core utility functions for the application.

#### Functions

##### `cn(...inputs: ClassValue[])`
Utility function for merging class names with Tailwind CSS classes.

```typescript
import { cn } from '@/lib/utils';

// Usage
const className = cn(
  'base-class',
  'additional-class',
  { 'conditional-class': condition },
  props.className
);
```

### i18n Library
**File**: `lib/i18n.ts`

Internationalization utilities for Turkish/English support.

#### Types
```typescript
export type Locale = 'tr' | 'en';
```

#### Hook: `useTranslation()`
Main hook for translation functionality.

#### Returns
```typescript
{
  t: (key: string) => string;           // Translation function
  locale: Locale;                       // Current locale
  changeLocale: (newLocale: Locale) => void; // Locale changer
}
```

#### Usage
```typescript
import { useTranslation } from '@/lib/i18n';

function MyComponent() {
  const { t, locale, changeLocale } = useTranslation();
  
  return (
    <div>
      <h1>{t('welcome.title')}</h1>
      <p>Current locale: {locale}</p>
      <button onClick={() => changeLocale('en')}>
        Switch to English
      </button>
    </div>
  );
}
```

### Markdown Library
**File**: `lib/markdown.ts`

Utilities for processing markdown files.

#### Interfaces
```typescript
export interface MarkdownFrontmatter {
  title?: string;
  description?: string;
  date?: string;
  [key: string]: any;
}
```

#### Functions

##### `getPostBySlug(filePath: string)`
Retrieves markdown content and frontmatter by file path.

```typescript
import { getPostBySlug } from '@/lib/markdown';

const { rawContent, frontmatter } = getPostBySlug('my-post');
```

##### `getAllMarkdownSlugsInDirectory(directoryPath: string)`
Gets all markdown file slugs in a directory.

```typescript
import { getAllMarkdownSlugsInDirectory } from '@/lib/markdown';

const slugs = getAllMarkdownSlugsInDirectory('blog');
```

### MDX Library
**File**: `lib/mdx.ts`

Advanced content processing for blog posts and topics.

#### Interfaces
```typescript
export interface BlogPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  author: string;
  category: string;
  tags: string[];
  image?: string;
  content?: string;
  topicPath?: string | null;
  isTypescriptPage?: boolean;
}
```

#### Functions

##### `getAllPosts(): Promise<BlogPost[]>`
Retrieves all blog posts from various directories.

```typescript
import { getAllPosts } from '@/lib/mdx';

const posts = await getAllPosts();
```

##### `getPostBySlug(slug: string): Promise<BlogPost>`
Retrieves a specific blog post by slug.

```typescript
import { getPostBySlug } from '@/lib/mdx';

const post = await getPostBySlug('my-blog-post');
```

### SEO Library
**File**: `lib/seo.ts`

SEO utilities and metadata generation.

#### Functions

##### `createPageMetadata(options)`
Creates page-specific metadata.

```typescript
import { createPageMetadata } from '@/lib/seo';

const metadata = createPageMetadata({
  title: 'Page Title',
  description: 'Page description',
  path: '/page-path',
  keywords: ['keyword1', 'keyword2'],
  imageUrl: '/image.jpg'
});
```

##### `createTopicPageMetadata(options)`
Creates topic-specific metadata.

```typescript
import { createTopicPageMetadata } from '@/lib/seo';

const metadata = createTopicPageMetadata({
  topicName: 'Python Programming',
  topicSlug: 'python',
  description: 'Learn Python programming',
  keywords: ['python', 'programming']
});
```

##### `createCourseSchema(course)`
Creates structured data for courses.

```typescript
import { createCourseSchema } from '@/lib/seo';

const schema = createCourseSchema({
  name: 'Python Course',
  description: 'Learn Python programming',
  provider: 'Kodleon',
  url: 'https://kodleon.com/python',
  instructor: 'Instructor Name'
});
```

##### `createArticleSchema(article)`
Creates structured data for articles.

```typescript
import { createArticleSchema } from '@/lib/seo';

const schema = createArticleSchema({
  headline: 'Article Title',
  description: 'Article description',
  authorName: 'Author Name',
  publishDate: '2024-01-01',
  url: 'https://kodleon.com/article'
});
```

##### `createFAQSchema(questions)`
Creates structured data for FAQ pages.

```typescript
import { createFAQSchema } from '@/lib/seo';

const schema = createFAQSchema([
  {
    question: 'What is Python?',
    answer: 'Python is a programming language.'
  }
]);
```

## Custom Hooks

### useToast Hook
**File**: `hooks/use-toast.ts`

Toast notification management hook.

#### Returns
```typescript
{
  toasts: ToasterToast[];
  toast: (props: Toast) => { id: string; dismiss: () => void; update: (props: ToasterToast) => void };
  dismiss: (toastId?: string) => void;
}
```

#### Usage
```typescript
import { useToast } from '@/hooks/use-toast';

function MyComponent() {
  const { toast } = useToast();
  
  const showToast = () => {
    toast({
      title: 'Success!',
      description: 'Operation completed successfully.',
    });
  };
  
  return <button onClick={showToast}>Show Toast</button>;
}
```

#### Toast Function
```typescript
// Basic toast
toast({
  title: 'Title',
  description: 'Description'
});

// Toast with action
toast({
  title: 'Title',
  description: 'Description',
  action: <Button>Action</Button>
});

// Different variants
toast({
  title: 'Error',
  description: 'Something went wrong',
  variant: 'destructive'
});
```

### Toaster Component
**File**: `components/ui/toaster.tsx`

Toast container component that displays toast notifications.

#### Usage
```typescript
import { Toaster } from '@/components/ui/toaster';

// Add to your app layout
<Toaster />
```

## Page Components

### Root Layout
**File**: `app/layout.tsx`

Main application layout with theme provider, navbar, and footer.

#### Features
- SEO metadata configuration
- Theme provider setup
- Font loading (Inter)
- Structured data (JSON-LD)
- Icon configuration
- Client-side redirects

### Homepage
**File**: `app/page.tsx`

Homepage component with SEO metadata.

#### Features
- Comprehensive SEO metadata
- Open Graph configuration
- Twitter Card setup
- Structured data

## Types and Interfaces

### Common Types
```typescript
// Locale type for internationalization
type Locale = 'tr' | 'en';

// Markdown frontmatter interface
interface MarkdownFrontmatter {
  title?: string;
  description?: string;
  date?: string;
  [key: string]: any;
}

// Blog post interface
interface BlogPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  author: string;
  category: string;
  tags: string[];
  image?: string;
  content?: string;
  topicPath?: string | null;
  isTypescriptPage?: boolean;
}

// Topic card props
interface TopicCardProps {
  title: string;
  description: string;
  href: string;
}

// Button variant props
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  asChild?: boolean;
}
```

## Usage Examples

### Building a Complete Page
```typescript
// app/my-page/page.tsx
import { Metadata } from 'next';
import { createPageMetadata } from '@/lib/seo';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';

export const metadata: Metadata = createPageMetadata({
  title: 'My Page',
  description: 'Page description',
  path: '/my-page'
});

export default function MyPage() {
  return (
    <div className="container mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>Page Title</CardTitle>
        </CardHeader>
        <CardContent>
          <MarkdownContent content="# Hello World" />
          <Button>Call to Action</Button>
        </CardContent>
      </Card>
    </div>
  );
}
```

### Creating a Form
```typescript
'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

const formSchema = z.object({
  email: z.string().email(),
  name: z.string().min(2),
});

export default function ContactForm() {
  const { toast } = useToast();
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: '',
      name: '',
    },
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    toast({
      title: 'Form submitted!',
      description: `Hello ${values.name}, we'll contact you at ${values.email}`,
    });
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Name</FormLabel>
              <FormControl>
                <Input placeholder="Your name" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="email"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Email</FormLabel>
              <FormControl>
                <Input placeholder="your@email.com" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit">Submit</Button>
      </form>
    </Form>
  );
}
```

### Implementing Internationalization
```typescript
'use client';

import { useTranslation } from '@/lib/i18n';
import { Button } from '@/components/ui/button';
import { LanguageSwitcher } from '@/components/LanguageSwitcher';

export default function MultilingualComponent() {
  const { t, locale } = useTranslation();

  return (
    <div>
      <h1>{t('welcome.title')}</h1>
      <p>{t('welcome.description')}</p>
      <p>Current language: {locale}</p>
      <LanguageSwitcher />
    </div>
  );
}
```

### Working with Markdown Content
```typescript
import { getPostBySlug } from '@/lib/markdown';
import MarkdownContent from '@/components/MarkdownContent';

export default async function BlogPost({ 
  params 
}: { 
  params: { slug: string } 
}) {
  const { rawContent, frontmatter } = getPostBySlug(params.slug);

  return (
    <article>
      <header>
        <h1>{frontmatter.title}</h1>
        <p>{frontmatter.description}</p>
      </header>
      <MarkdownContent content={rawContent} />
    </article>
  );
}
```

This documentation covers all the major public APIs, components, and utilities in the Kodleon platform. Each component and function includes proper TypeScript interfaces, usage examples, and implementation details to help developers effectively use and extend the platform.