'use client';

import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface MarkdownContentProps {
  content: string;
}

export default function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        img: ({ node, ...props }) => (
          <img 
            className="rounded-lg shadow-md my-6 mx-auto max-w-full h-auto" 
            {...props} 
            loading="lazy"
          />
        ),
        h1: ({ node, ...props }) => (
          <h1 className="text-4xl font-bold mt-8 mb-6 text-gray-800 dark:text-gray-100" {...props} />
        ),
        h2: ({ node, ...props }) => (
          <h2 className="text-3xl font-semibold mt-8 mb-4 text-gray-800 dark:text-gray-100 border-b pb-2 border-gray-200 dark:border-gray-700" {...props} />
        ),
        h3: ({ node, ...props }) => (
          <h3 className="text-2xl font-semibold mt-6 mb-3 text-gray-800 dark:text-gray-100" {...props} />
        ),
        p: ({ node, ...props }) => (
          <p className="text-lg text-gray-700 dark:text-gray-300 mb-4 leading-relaxed" {...props} />
        ),
        ul: ({ node, ...props }) => (
          <ul className="list-disc pl-6 mb-4 text-gray-700 dark:text-gray-300" {...props} />
        ),
        ol: ({ node, ...props }) => (
          <ol className="list-decimal pl-6 mb-4 text-gray-700 dark:text-gray-300" {...props} />
        ),
        li: ({ node, ...props }) => (
          <li className="mb-2 text-lg" {...props} />
        ),
        a: ({ node, ...props }) => (
          <a className="text-blue-600 dark:text-blue-400 hover:underline" {...props} />
        ),
        blockquote: ({ node, ...props }) => (
          <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-4 py-2 italic text-gray-600 dark:text-gray-400 my-4" {...props} />
        ),
        code: ({ node, inline, className, children, ...props }: { node?: any; inline?: boolean; className?: string; children?: React.ReactNode }) => (
          inline ? 
            <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-sm font-mono" {...props}>{children}</code> :
            <code className="block bg-gray-100 dark:bg-gray-800 p-4 rounded-lg text-sm font-mono overflow-x-auto my-4" {...props}>{children}</code>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
} 