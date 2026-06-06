'use client';

import React, { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Message } from '@/types';

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

export default function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className="message">
      {/* Avatar */}
      <div className={`message-avatar ${message.role}`}>
        {isUser ? (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        )}
      </div>

      {/* Body */}
      <div className="message-body">
        <div className="message-sender">
          {isUser ? 'You' : 'RAG Assistant'}
        </div>

        <div className={`message-content ${isStreaming ? 'streaming-cursor' : ''}`}>
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{
                pre({ children, ...props }) {
                  return <CodeBlock {...props}>{children}</CodeBlock>;
                },
                code({ children, className, ...props }) {
                  const isInline = !className;
                  if (isInline) {
                    return <code {...props}>{children}</code>;
                  }
                  return <code className={className} {...props}>{children}</code>;
                },
              }}
            >
              {message.content || '...'}
            </ReactMarkdown>
          )}
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && !isStreaming && (
          <div className="sources-list">
            {message.sources.map((source, i) => (
              <span key={i} className="source-chip">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
                {source.source} ({(source.score * 100).toFixed(0)}%)
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Code Block with Copy Button ───────────────────────────────

function CodeBlock({ children, ...props }: any) {
  const [copied, setCopied] = useState(false);

  // Extract language from the code element's className
  const codeElement = React.Children.toArray(children).find(
    (child: any) => child?.type === 'code'
  ) as any;

  const className = (codeElement?.props?.className as string) || '';
  const lang = className.replace('hljs ', '').replace('language-', '') || 'code';
  const codeText = codeElement?.props?.children
    ? String(codeElement.props.children).replace(/\n$/, '')
    : '';

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(codeText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [codeText]);

  return (
    <div>
      <div className="code-block-header">
        <span className="lang">{lang}</span>
        <button className="copy-btn" onClick={handleCopy}>
          {copied ? (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              Copied
            </>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
      <pre {...props}>{children}</pre>
    </div>
  );
}
