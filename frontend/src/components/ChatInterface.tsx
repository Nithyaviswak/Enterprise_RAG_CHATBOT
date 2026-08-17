'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import ChatMessage from './ChatMessage';
import DevPanel from './DevPanel';
import FileUpload from './FileUpload';
import GraphExplorer from './GraphExplorer';
import ThemeToggle from './ThemeToggle';
import { Message, DebugInfo } from '@/types';

interface ChatInterfaceProps {
  messages: Message[];
  isStreaming: boolean;
  onSendMessage: (content: string) => void;
  onStopStreaming: () => void;
  onToggleSidebar: () => void;
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  debugMode?: boolean;
  lastDebug?: DebugInfo | null;
  onToggleDebug?: () => void;
}

const SUGGESTIONS = [
  { label: 'Explain', text: 'Explain the key findings from the uploaded documents' },
  { label: 'Summarize', text: 'Summarize all documents in the knowledge base' },
  { label: 'Analyze', text: 'What are the main themes across my documents?' },
  { label: 'Search', text: 'Find information about a specific topic in my files' },
];

export default function ChatInterface({
  messages,
  isStreaming,
  onSendMessage,
  onStopStreaming,
  onToggleSidebar,
  theme,
  onToggleTheme,
  debugMode = false,
  lastDebug = null,
  onToggleDebug,
}: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [showUpload, setShowUpload] = useState(false);
  const [showGraph, setShowGraph] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSend = useCallback(() => {
    if (!input.trim() || isStreaming) return;
    onSendMessage(input.trim());
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [input, isStreaming, onSendMessage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleSuggestion = useCallback((text: string) => {
    onSendMessage(text);
  }, [onSendMessage]);

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-main">
      {/* Header */}
      <header className="chat-header">
        <div className="chat-header-left">
          <button className="toggle-sidebar-btn" onClick={onToggleSidebar} title="Toggle sidebar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <line x1="9" y1="3" x2="9" y2="21"/>
            </svg>
          </button>
          <div className="model-badge">
            <span className="dot" />
            Gemini 2.5 Flash
          </div>
        </div>
        <div className="chat-header-right">
          <button
            className={`icon-btn ${debugMode ? 'active' : ''}`}
            onClick={onToggleDebug}
            title={debugMode ? 'Disable developer mode (returns retrieval internals)' : 'Enable developer mode (returns retrieval internals)'}
            style={debugMode ? { color: 'var(--color-accent)' } : undefined}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="16 18 22 12 16 6" />
              <polyline points="8 6 2 12 8 18" />
            </svg>
          </button>
          <button
            className="icon-btn"
            onClick={() => setShowGraph(true)}
            title="Knowledge Graph"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <circle cx="19" cy="5" r="2" />
              <circle cx="5" cy="19" r="2" />
              <circle cx="19" cy="19" r="2" />
              <line x1="12" y1="9" x2="12" y2="5" />
              <line x1="14.5" y1="14.5" x2="17.5" y2="17.5" />
              <line x1="9.5" y1="14.5" x2="6.5" y2="17.5" />
            </svg>
          </button>
          <button
            className="icon-btn"
            onClick={() => setShowUpload(true)}
            title="Upload document"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </button>
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
        </div>
      </header>

      {/* Developer insights panel */}
      {debugMode && <DevPanel debug={lastDebug} />}

      {/* Messages or Welcome Screen */}
      <div className="messages-container">
        {isEmpty ? (
          <div className="welcome-screen">
            <div className="welcome-logo">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </div>
            <h1>RAG Chatbot</h1>
            <p>
              Ask questions about your documents. Upload files to build your
              knowledge base, then chat with AI-powered retrieval.
            </p>
            <div className="prompt-suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  className="prompt-suggestion"
                  onClick={() => handleSuggestion(s.text)}
                >
                  <div className="label">{s.label}</div>
                  <div className="text">{s.text}</div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-inner">
            {messages.map((msg, i) => (
              <ChatMessage
                key={msg.id}
                message={msg}
                isStreaming={
                  isStreaming &&
                  msg.role === 'assistant' &&
                  i === messages.length - 1
                }
              />
            ))}
            {isStreaming && messages[messages.length - 1]?.role !== 'assistant' && (
              <div className="message">
                <div className="message-avatar assistant">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                  </svg>
                </div>
                <div className="message-body">
                  <div className="message-sender">RAG Assistant</div>
                  <div className="typing-indicator">
                    <div className="dot" />
                    <div className="dot" />
                    <div className="dot" />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="input-area">
        <div className="input-wrapper">
          <div className="input-container">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about your documents..."
              rows={1}
              disabled={isStreaming}
            />
            <div className="input-actions">
              <button
                className="input-action-btn"
                onClick={() => setShowUpload(true)}
                title="Attach file"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                </svg>
              </button>
              {isStreaming ? (
                <button
                  className="input-action-btn"
                  onClick={onStopStreaming}
                  title="Stop generating"
                  style={{ color: 'var(--color-error)' }}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="6" width="12" height="12" rx="2" />
                  </svg>
                </button>
              ) : (
                <button
                  className="input-action-btn send-btn"
                  onClick={handleSend}
                  disabled={!input.trim()}
                  title="Send message"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                </button>
              )}
            </div>
          </div>
          <div className="input-footer">
            RAG Chatbot uses Gemini AI with document retrieval. Responses may not always be accurate.
          </div>
        </div>
      </div>

      {/* File Upload Modal */}
      <FileUpload
        isOpen={showUpload}
        onClose={() => setShowUpload(false)}
      />

      {/* Knowledge Graph Explorer */}
      <GraphExplorer
        isOpen={showGraph}
        onClose={() => setShowGraph(false)}
      />
    </div>
  );
}
