/**
 * API Client — Backend Communication Layer
 *
 * Handles REST API calls and SSE streaming connections
 * to the FastAPI backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new ApiError(errorBody || response.statusText, response.status);
  }

  return response.json();
}

// ── Chat API ──────────────────────────────────────────────────

export async function sendMessage(
  message: string,
  conversationId: string | null,
  signal?: AbortSignal
): Promise<Response> {
  const url = `${API_BASE}/api/chat`;
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
      use_ragflow: true,
    }),
    signal,
  });
}

export async function getConversations() {
  return request<{ conversations: any[] }>('/api/chat/history');
}

export async function getConversationMessages(conversationId: string) {
  return request<{ messages: any[] }>(`/api/chat/${conversationId}`);
}

export async function deleteConversation(conversationId: string) {
  return request<{ status: string }>(`/api/chat/${conversationId}`, {
    method: 'DELETE',
  });
}

// ── Document API ──────────────────────────────────────────────

export async function uploadDocument(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);

  const url = `${API_BASE}/api/documents/upload`;
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new ApiError('Upload failed', response.status);
  }

  return response.json();
}

export async function getDocuments() {
  return request<{ documents: any[] }>('/api/documents');
}

export async function deleteDocument(documentId: string) {
  return request<{ status: string }>(`/api/documents/${documentId}`, {
    method: 'DELETE',
  });
}

// ── Health API ────────────────────────────────────────────────

export async function healthCheck() {
  return request<{ status: string; version: string; services: Record<string, string> }>(
    '/api/health'
  );
}

// ── Knowledge Graph API ────────────────────────────────────────

export async function getGraphStats() {
  return request<{ node_count: number; edge_count: number; entity_type_counts: Record<string, number>; relation_type_counts: Record<string, number>; documents_processed: number }>('/api/graph/stats');
}

export async function getGraphEntities(params?: { entity_type?: string; limit?: number; offset?: number; search?: string }) {
  const query = new URLSearchParams();
  if (params?.entity_type) query.set('entity_type', params.entity_type);
  if (params?.limit) query.set('limit', String(params.limit));
  if (params?.offset) query.set('offset', String(params.offset));
  if (params?.search) query.set('search', params.search);
  return request<{ entities: any[]; total: number }>(`/api/graph/entities?${query}`);
}

export async function getGraphEntityDetail(entityId: string) {
  return request<any>(`/api/graph/entities/${entityId}`);
}

export async function exploreGraph(entityId: string, maxHops = 2, maxNodes = 50) {
  return request<{ nodes: any[]; edges: any[]; central_entity: any }>(
    `/api/graph/entities/${entityId}/neighborhood?max_hops=${maxHops}&max_nodes=${maxNodes}`
  );
}

export async function searchGraph(query: string, entityTypes?: string[], useHybrid = true) {
  return request<{ entities: any[]; relationships: any[]; paths: any[][]; vector_results: any[]; query_type: string }>(
    '/api/graph/search',
    {
      method: 'POST',
      body: JSON.stringify({
        query,
        entity_types: entityTypes || null,
        max_hops: 2,
        top_k: 10,
        use_hybrid_search: useHybrid,
      }),
    }
  );
}

export async function indexDocumentToGraph(documentId: string) {
  return request<{ message: string; entities_created: number; relationships_created: number; source: string }>(
    `/api/graph/index-document?document_id=${documentId}`,
    { method: 'POST' }
  );
}

export async function deleteAllGraphData() {
  return request<{ status: string }>('/api/graph/clear', { method: 'DELETE' });
}

// ── SSE Stream Parser ─────────────────────────────────────────

export async function* parseSSEStream(
  response: Response
): AsyncGenerator<{ type: string; [key: string]: any }> {
  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            yield data;
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
