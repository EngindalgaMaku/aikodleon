'use client';

import ReactMarkdown from 'react-markdown';
import { content } from './content';
import { content_part2 } from './content_part2';
import { content_part3 } from './content_part3';
import { content_part4 } from './content_part4';
import { content_part5 } from './content_part5';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import remarkGfm from 'remark-gfm';
import Link from 'next/link';

const CustomAdmonition = ({ type, children }: { type: string; children: React.ReactNode }) => {
  const styles = {
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    warning: 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800',
    tip: 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800'
  };

  return (
    <div className={`my-8 p-6 rounded-lg border ${styles[type as keyof typeof styles]}`}>
      <div className="prose dark:prose-invert max-w-none">
        {children}
      </div>
    </div>
  );
};

export default function Page() {
  // Combine all content parts
  const fullContent = content + content_part2 + content_part3 + content_part4 + content_part5;
  
  return (
    <div className="max-w-5xl mx-auto py-8">
      <article className="prose prose-lg dark:prose-invert max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({children}) => <h1 className="text-4xl font-bold mb-8">{children}</h1>,
            h2: ({children}) => <h2 className="text-3xl font-semibold mt-12 mb-6">{children}</h2>,
            h3: ({children}) => <h3 className="text-2xl font-semibold mt-8 mb-4">{children}</h3>,
            p: ({children}) => <p className="mb-4 leading-relaxed">{children}</p>,
            ul: ({children}) => <ul className="list-disc pl-6 mb-6 space-y-2">{children}</ul>,
            ol: ({children}) => <ol className="list-decimal pl-6 mb-6 space-y-2">{children}</ol>,
            li: ({children}) => <li className="mb-1">{children}</li>,
            a: ({href, children}) => (
              <Link href={href || '#'} className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-200">
                {children}
              </Link>
            ),
            code({className, children}) {
              const match = /language-(\w+)/.exec(className || '');
              const language = match ? match[1] : '';
              const code = String(children).replace(/\n$/, '');
              
              if (!className) {
                return <code className="bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded-md text-sm font-mono">{children}</code>;
              }

              return (
                <div className="my-6">
                  <SyntaxHighlighter
                    language={language}
                    style={vscDarkPlus}
                    className="rounded-lg !bg-slate-900"
                    PreTag="div"
                  >
                    {code}
                  </SyntaxHighlighter>
                </div>
              );
            },
            blockquote({children}) {
              const text = String(children);
              if (text.startsWith('::: info')) {
                return <CustomAdmonition type="info">{children}</CustomAdmonition>;
              }
              if (text.startsWith('::: warning')) {
                return <CustomAdmonition type="warning">{children}</CustomAdmonition>;
              }
              if (text.startsWith('::: tip')) {
                return <CustomAdmonition type="tip">{children}</CustomAdmonition>;
              }
              return <blockquote className="border-l-4 border-slate-300 dark:border-slate-700 pl-4 my-4">{children}</blockquote>;
            }
          }}
        >
          {fullContent}
        </ReactMarkdown>
      </article>
    </div>
  );
} 