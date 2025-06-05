'use client';

import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import { content } from './content';
import CodeRunner from '../siniflar-ve-nesneler/components/CodeRunner';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import remarkGfm from 'remark-gfm';

export default function Page() {
  return (
    <div className="max-w-5xl mx-auto py-8">
      <article className="prose prose-lg dark:prose-invert max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({className, children}) {
              const match = /language-(\w+)/.exec(className || '');
              const language = match ? match[1] : '';
              const code = String(children).replace(/\n$/, '');
              
              if (!className) {
                return <code className="bg-muted px-1.5 py-0.5 rounded-md">{children}</code>;
              }

              return language === 'python' ? (
                <div className="my-4">
                  <CodeRunner initialCode={code} />
                </div>
              ) : (
                <SyntaxHighlighter
                  language={language}
                  style={vscDarkPlus}
                  PreTag="div"
                >
                  {code}
                </SyntaxHighlighter>
              );
            },
            div({className, children}) {
              if (className?.includes('info')) {
                return (
                  <div className="bg-blue-50 dark:bg-blue-900/10 p-6 rounded-lg my-6">
                    {children}
                  </div>
                );
              }
              if (className?.includes('warning')) {
                return (
                  <div className="bg-yellow-50 dark:bg-yellow-900/10 p-6 rounded-lg my-6">
                    {children}
                  </div>
                );
              }
              if (className?.includes('tip')) {
                return (
                  <div className="bg-purple-50 dark:bg-purple-900/10 p-6 rounded-lg my-6">
                    {children}
                  </div>
                );
              }
              return <div>{children}</div>;
            }
          }}
        >
          {content}
        </ReactMarkdown>
      </article>
    </div>
  );
} 