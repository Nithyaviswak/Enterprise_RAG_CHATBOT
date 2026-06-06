'use client';

import { useState, useCallback, useRef } from 'react';
import { Message, Source, Conversation } from '@/types';
import {
  sendMessage as apiSendMessage,
  parseSSEStream,
  getConversations,
  getConversationMessages,
  deleteConversation as apiDeleteConversation,
} from '@/lib/api';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [sources, setSources] = useState<Source[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Load conversations list
  const loadConversations = useCallback(async () => {
    try {
      const data = await getConversations();
      setConversations(data.conversations || []);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  }, []);

  // Load messages for a conversation
  const loadConversation = useCallback(async (conversationId: string) => {
    try {
      const data = await getConversationMessages(conversationId);
      const msgs: Message[] = (data.messages || []).map((m: any) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        sources: typeof m.sources === 'string' ? JSON.parse(m.sources) : m.sources || [],
        created_at: m.created_at,
      }));
      setMessages(msgs);
      setActiveConversationId(conversationId);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  }, []);

  // Send a message
  const sendMessage = useCallback(async (content: string) => {
    if (isStreaming || !content.trim()) return;

    // Add user message to UI immediately
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      sources: [],
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Create placeholder for assistant response
    const assistantId = `assistant-${Date.now()}`;
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, assistantMessage]);

    setIsStreaming(true);
    setSources([]);

    // Create abort controller
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const response = await apiSendMessage(
        content,
        activeConversationId,
        abortController.signal
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      let currentSources: Source[] = [];

      for await (const event of parseSSEStream(response)) {
        switch (event.type) {
          case 'metadata':
            if (event.conversation_id && !activeConversationId) {
              setActiveConversationId(event.conversation_id);
            }
            currentSources = event.sources || [];
            setSources(currentSources);
            break;

          case 'token':
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + (event.content || '') }
                  : m
              )
            );
            break;

          case 'title':
            // Update conversation title in sidebar
            if (event.title) {
              setConversations((prev) =>
                prev.map((c) =>
                  c.id === activeConversationId
                    ? { ...c, title: event.title! }
                    : c
                )
              );
              await loadConversations();
            }
            break;

          case 'done':
            // Attach sources to the assistant message
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, sources: currentSources } : m
              )
            );
            break;

          case 'error':
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: `⚠️ Error: ${event.message || 'Unknown error'}` }
                  : m
              )
            );
            break;
        }
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `⚠️ Connection error: ${error.message}` }
              : m
          )
        );
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
      // Refresh conversation list
      await loadConversations();
    }
  }, [isStreaming, activeConversationId, loadConversations]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort();
    setIsStreaming(false);
  }, []);

  // New chat
  const newChat = useCallback(() => {
    setMessages([]);
    setActiveConversationId(null);
    setSources([]);
  }, []);

  // Delete conversation
  const deleteConversation = useCallback(async (conversationId: string) => {
    try {
      await apiDeleteConversation(conversationId);
      if (activeConversationId === conversationId) {
        newChat();
      }
      await loadConversations();
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  }, [activeConversationId, newChat, loadConversations]);

  return {
    messages,
    conversations,
    activeConversationId,
    isStreaming,
    sources,
    sendMessage,
    stopStreaming,
    newChat,
    loadConversations,
    loadConversation,
    deleteConversation,
  };
}
