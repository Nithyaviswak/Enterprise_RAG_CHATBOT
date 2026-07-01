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

// ── Knowledge Graph Types ──────────────────────────────────────

export interface GraphEntity {
  id: string;
  name: string;
  entity_type: string;
  description: string;
  aliases: string[];
  source_document: string;
  created_at: string;
}

export interface GraphRelationship {
  id: string;
  source_name: string;
  target_name: string;
  relation_type: string;
  description: string;
  weight: number;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  description: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  weight: number;
}

export interface GraphExploreResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  central_entity: GraphEntity;
}

export interface GraphSearchResponse {
  entities: GraphEntity[];
  relationships: GraphRelationship[];
  paths: any[][];
  vector_results: any[];
  query_type: string;
}

export interface GraphStats {
  node_count: number;
  edge_count: number;
  entity_type_counts: Record<string, number>;
  relation_type_counts: Record<string, number>;
  documents_processed: number;
}

export interface EntityDetail {
  entity: GraphEntity;
  relationships: GraphRelationship[];
  related_entities: GraphEntity[];
  contexts: string[];
}
