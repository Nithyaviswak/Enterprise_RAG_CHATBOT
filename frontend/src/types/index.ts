export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: Source[];
  created_at: string;
  meta?: AssistantMeta | null;
}

export interface Source {
  source: string;
  score: number;
  page?: number | null;
  id?: string | null;
}

export interface Confidence {
  overall_confidence: number;
  confidence_level: 'low' | 'medium' | 'high';
  retrieval_confidence: number;
  grounding_ratio: number;
  risky?: boolean;
}

export interface AssistantMeta {
  request_id: string;
  latency_ms: number | null;
  confidence: Confidence | null;
  failure_type: string | null;
  refused: boolean;
  answered: boolean;
  hallucination?: { risk_level: string; grounded_ratio: number } | null;
}

export interface DebugInfo {
  query: string;
  request_id: string;
  retrieved_documents: Array<{
    source: string;
    page?: number | null;
    score?: number;
    rerank_score?: number | null;
    retrieval_method?: string;
    excerpt?: string;
  }>;
  retrieval_confidence: number | null;
  retrieval_methods: string[];
  similarity_scores: number[];
  reranking_scores: number[];
  final_context: string[];
  system_prompt?: string | null;
  confidence: Confidence | null;
  failure_type: string | null;
  stage_times: Record<string, { latency_ms: number; status: string; [key: string]: any }>;
  timestamp: string;
}

export interface LiveMetrics {
  total_requests: number;
  avg_retrieval_latency_ms: number | null;
  avg_generation_latency_ms: number | null;
  avg_total_latency_ms: number | null;
  avg_retrieval_confidence: number | null;
  avg_grounding_ratio: number | null;
  refusal_rate: number;
  hallucination_risk_rate: number;
  failure_counts: Record<string, number>;
  recent: Record<string, any>[];
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatStreamEvent {
  type: 'metadata' | 'token' | 'title' | 'done' | 'error' | 'debug';
  content?: string;
  conversation_id?: string;
  sources?: Source[];
  confidence?: Confidence;
  request_id?: string;
  latency_ms?: number | null;
  failure_type?: string | null;
  refused?: boolean;
  answered?: boolean;
  debug?: DebugInfo;
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
