'use client';

import { useState, useEffect } from 'react';
import ChatSidebar from '@/components/ChatSidebar';
import ChatInterface from '@/components/ChatInterface';
import ThemeToggle from '@/components/ThemeToggle';
import { useChat } from '@/hooks/useChat';
import { useTheme } from '@/hooks/useTheme';

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { theme, toggleTheme } = useTheme();

  const {
    messages,
    conversations,
    activeConversationId,
    isStreaming,
    debugMode,
    lastDebug,
    toggleDebugMode,
    sendMessage,
    stopStreaming,
    newChat,
    loadConversations,
    loadConversation,
    deleteConversation,
  } = useChat();

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <ChatSidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((prev) => !prev)}
        onNewChat={newChat}
        onSelectConversation={loadConversation}
        onDeleteConversation={deleteConversation}
      />

      {/* Main Chat Area */}
      <ChatInterface
        messages={messages}
        isStreaming={isStreaming}
        onSendMessage={sendMessage}
        onStopStreaming={stopStreaming}
        onToggleSidebar={() => setSidebarCollapsed((prev) => !prev)}
        theme={theme}
        onToggleTheme={toggleTheme}
        debugMode={debugMode}
        lastDebug={lastDebug}
        onToggleDebug={toggleDebugMode}
      />
    </div>
  );
}
