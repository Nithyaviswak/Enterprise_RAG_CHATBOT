export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: Source[];
  created_at: string;
}

export interface Source {
  source: string;
  score: number;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatStreamEvent {
  type: 'metadata' | 'token' | 'title' | 'done' | 'error';
  content?: string;
  conversation_id?: string;
  sources?: Source[];
  title?: string;
  message?: string;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  file_type: string;
  size_bytes: number;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  chunks_count: number;
  created_at: string;
}
