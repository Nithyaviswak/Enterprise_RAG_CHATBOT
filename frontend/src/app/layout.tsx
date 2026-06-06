import type { Metadata } from 'next';
import './globals.css';
import 'highlight.js/styles/github-dark.css';

export const metadata: Metadata = {
  title: 'RAG Chatbot — AI-Powered Document Assistant',
  description:
    'Production-grade RAG chatbot powered by Google Gemini, RAGFlow, and fine-tuned embeddings. Chat with your documents using AI.',
  keywords: ['RAG', 'chatbot', 'AI', 'Gemini', 'document', 'retrieval'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#0a0a0f" />
      </head>
      <body>{children}</body>
    </html>
  );
}
