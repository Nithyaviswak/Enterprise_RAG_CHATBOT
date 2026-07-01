'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { GraphNode, GraphEdge } from '@/types';
import * as api from '@/lib/api';

interface GraphExplorerProps {
  isOpen: boolean;
  onClose: () => void;
}

interface LayoutNode extends GraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

const TYPE_COLORS: Record<string, string> = {
  Organization: '#4f46e5',
  Person: '#059669',
  Product: '#d97706',
  Location: '#dc2626',
  Event: '#7c3aed',
  Document: '#0891b2',
  Concept: '#65a30d',
  Date: '#ca8a04',
  Technology: '#2563eb',
};

const TYPE_ICONS: Record<string, string> = {
  Organization: '🏢',
  Person: '👤',
  Product: '📦',
  Location: '📍',
  Event: '📅',
  Document: '📄',
  Concept: '💡',
  Date: '📆',
  Technology: '⚙️',
};

function simulateForce(
  nodes: LayoutNode[],
  edges: GraphEdge[],
  width: number,
  height: number
): LayoutNode[] {
  const k = 200;
  const iterations = 50;

  for (let iter = 0; iter < iterations; iter++) {
    const forces: { fx: number; fy: number }[] = nodes.map(() => ({ fx: 0, fy: 0 }));

    // Repulsion between all nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        let dx = nodes[j].x - nodes[i].x;
        let dy = nodes[j].y - nodes[i].y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = k * k / dist;
        forces[i].fx -= (force * dx) / dist;
        forces[i].fy -= (force * dy) / dist;
        forces[j].fx += (force * dx) / dist;
        forces[j].fy += (force * dy) / dist;
      }
    }

    // Attraction along edges
    for (const edge of edges) {
      const si = nodes.findIndex((n) => n.label === edge.source);
      const ti = nodes.findIndex((n) => n.label === edge.target);
      if (si === -1 || ti === -1) continue;
      let dx = nodes[ti].x - nodes[si].x;
      let dy = nodes[ti].y - nodes[si].y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = (dist * dist) / k;
      forces[si].fx += (force * dx) / dist;
      forces[si].fy += (force * dy) / dist;
      forces[ti].fx -= (force * dx) / dist;
      forces[ti].fy -= (force * dy) / dist;
    }

    // Apply forces
    for (let i = 0; i < nodes.length; i++) {
      nodes[i].vx = (nodes[i].vx || 0) * 0.5 + forces[i].fx * 0.1;
      nodes[i].vy = (nodes[i].vy || 0) * 0.5 + forces[i].fy * 0.1;
      nodes[i].x += nodes[i].vx;
      nodes[i].y += nodes[i].vy;

      // Keep within bounds
      nodes[i].x = Math.max(30, Math.min(width - 30, nodes[i].x));
      nodes[i].y = Math.max(30, Math.min(height - 30, nodes[i].y));
    }
  }

  return nodes;
}

export default function GraphExplorer({ isOpen, onClose }: GraphExplorerProps) {
  const [nodes, setNodes] = useState<LayoutNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState<LayoutNode | null>(null);
  const [entityTypes, setEntityTypes] = useState<string[]>([]);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Load stats on mount
  useEffect(() => {
    if (!isOpen) return;
    loadStats();
    loadGraph(null);
  }, [isOpen]);

  const loadStats = async () => {
    try {
      const s = await api.getGraphStats();
      setStats(s);
      setEntityTypes(Object.keys(s.entity_type_counts || {}));
    } catch {
      // Stats may not be available
    }
  };

  const loadGraph = useCallback(async (entityId: string | null) => {
    setLoading(true);
    setError(null);
    try {
      if (entityId) {
        const data = await api.exploreGraph(entityId, 2, 50);
        layoutGraph(data.nodes, data.edges);
      } else {
        const entities = await api.getGraphEntities({ limit: 30 });
        if (entities.entities.length === 0) {
          setNodes([]);
          setEdges([]);
          setLoading(false);
          return;
        }
        const firstId = entities.entities[0]?.id;
        if (firstId) {
          const data = await api.exploreGraph(firstId, 1, 50);
          layoutGraph(data.nodes, data.edges);
        }
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load graph');
      setLoading(false);
    }
  }, []);

  const layoutGraph = (rawNodes: GraphNode[], rawEdges: GraphEdge[]) => {
    const container = containerRef.current;
    const width = container?.clientWidth || 600;
    const height = container?.clientHeight || 500;

    const centerX = width / 2;
    const centerY = height / 2;

    const layoutNodes: LayoutNode[] = rawNodes.map((n, i) => ({
      ...n,
      x: centerX + (Math.random() - 0.5) * width * 0.5,
      y: centerY + (Math.random() - 0.5) * height * 0.5,
      vx: 0,
      vy: 0,
    }));

    const positioned = simulateForce(layoutNodes, rawEdges, width, height);
    setNodes(positioned);
    setEdges(rawEdges);
    setLoading(false);
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    try {
      const result = await api.searchGraph(searchQuery);
      if (result.entities.length > 0) {
        const firstId = result.entities[0]?.id;
        if (firstId) {
          const data = await api.exploreGraph(firstId, 2, 50);
          layoutGraph(data.nodes, data.edges);
        }
      }
    } catch (err: any) {
      setError(err.message || 'Search failed');
      setLoading(false);
    }
  };

  const handleNodeClick = (node: LayoutNode) => {
    setSelectedNode(node);
  };

  const handleNodeDoubleClick = async (node: LayoutNode) => {
    setLoading(true);
    try {
      const data = await api.exploreGraph(node.id, 2, 50);
      layoutGraph(data.nodes, data.edges);
    } catch (err: any) {
      setError(err.message || 'Failed to expand node');
      setLoading(false);
    }
  };

  const handleFilter = async (type: string | null) => {
    setActiveFilter(type);
    setLoading(true);
    try {
      const entities = await api.getGraphEntities({
        entity_type: type || undefined,
        limit: 30,
      });
      if (entities.entities.length > 0) {
        const firstId = entities.entities[0]?.id;
        if (firstId) {
          const data = await api.exploreGraph(firstId, 1, 50);
          layoutGraph(data.nodes, data.edges);
        }
      } else {
        setNodes([]);
        setEdges([]);
        setLoading(false);
      }
    } catch {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const filteredEdges = selectedNode
    ? edges.filter((e) => e.source === selectedNode.label || e.target === selectedNode.label)
    : edges;

  const filteredNodes = selectedNode
    ? nodes.filter((n) =>
        n.label === selectedNode.label ||
        filteredEdges.some((e) => e.source === n.label || e.target === n.label)
      )
    : nodes;

  return (
    <div className="graph-overlay" onClick={onClose}>
      <div className="graph-modal" onClick={(e) => e.stopPropagation()}>
        <div className="graph-header">
          <h2>Knowledge Graph Explorer</h2>
          <div className="graph-header-actions">
            {stats && (
              <span className="graph-stats-badge">
                {stats.node_count} nodes · {stats.edge_count} edges
              </span>
            )}
            <button className="graph-close-btn" onClick={onClose}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        <div className="graph-toolbar">
          <div className="graph-search">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search entities..."
            />
            <button onClick={handleSearch}>Search</button>
          </div>
          <div className="graph-filters">
            <button
              className={`graph-filter-btn ${activeFilter === null ? 'active' : ''}`}
              onClick={() => handleFilter(null)}
            >
              All
            </button>
            {entityTypes.map((type) => (
              <button
                key={type}
                className={`graph-filter-btn ${activeFilter === type ? 'active' : ''}`}
                onClick={() => handleFilter(type)}
              >
                {TYPE_ICONS[type] || '•'} {type}
              </button>
            ))}
          </div>
        </div>

        <div className="graph-content">
          <div className="graph-canvas-container" ref={containerRef}>
            {loading ? (
              <div className="graph-loading">
                <div className="graph-spinner" />
                <p>Loading knowledge graph...</p>
              </div>
            ) : error ? (
              <div className="graph-error">
                <p>⚠️ {error}</p>
                <button onClick={() => loadGraph(null)}>Retry</button>
              </div>
            ) : nodes.length === 0 ? (
              <div className="graph-empty">
                <p>No entities in the knowledge graph yet.</p>
                <p className="graph-hint">
                  Upload documents and index them to the graph to see entities here.
                </p>
              </div>
            ) : (
              <svg
                ref={svgRef}
                className="graph-svg"
                width="100%"
                height="100%"
                viewBox={`0 0 ${containerRef.current?.clientWidth || 600} ${containerRef.current?.clientHeight || 500}`}
              >
                <defs>
                  {filteredEdges.map((edge, i) => {
                    const source = filteredNodes.find((n) => n.label === edge.source);
                    const target = filteredNodes.find((n) => n.label === edge.target);
                    if (!source || !target) return null;
                    const dx = target.x - source.x;
                    const dy = target.y - source.y;
                    const len = Math.sqrt(dx * dx + dy * dy) || 1;
                    const mx = (source.x + target.x) / 2;
                    const my = (source.y + target.y) / 2;
                    const nx = -dy / len * 10;
                    const ny = dx / len * 10;
                    return (
                      <marker
                        key={`arrow-${i}`}
                        id={`arrow-${i}`}
                        viewBox="0 0 10 10"
                        refX="20"
                        refY="5"
                        markerWidth="6"
                        markerHeight="6"
                        orient="auto"
                      >
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-border)" />
                      </marker>
                    );
                  })}
                </defs>

                {/* Edges */}
                {filteredEdges.map((edge, i) => {
                  const source = filteredNodes.find((n) => n.label === edge.source);
                  const target = filteredNodes.find((n) => n.label === edge.target);
                  if (!source || !target) return null;
                  const isHighlighted = selectedNode &&
                    (edge.source === selectedNode.label || edge.target === selectedNode.label);
                  return (
                    <g key={`edge-${i}`}>
                      <line
                        x1={source.x}
                        y1={source.y}
                        x2={target.x}
                        y2={target.y}
                        stroke="var(--color-border)"
                        strokeWidth={isHighlighted ? 2 : 1}
                        strokeOpacity={isHighlighted ? 0.8 : 0.3}
                        markerEnd={`url(#arrow-${i})`}
                      />
                      <text
                        x={(source.x + target.x) / 2}
                        y={(source.y + target.y) / 2 - 6}
                        textAnchor="middle"
                        fill="var(--color-muted)"
                        fontSize="10"
                      >
                        {edge.label}
                      </text>
                    </g>
                  );
                })}

                {/* Nodes */}
                {filteredNodes.map((node) => {
                  const color = TYPE_COLORS[node.type] || '#6b7280';
                  const isSelected = selectedNode?.id === node.id;
                  const radius = isSelected ? 28 : node.type === 'Organization' ? 24 : 20;
                  return (
                    <g
                      key={node.id}
                      className="graph-node"
                      onClick={() => handleNodeClick(node)}
                      onDoubleClick={() => handleNodeDoubleClick(node)}
                      style={{ cursor: 'pointer' }}
                    >
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={radius}
                        fill={isSelected ? color : `${color}33`}
                        stroke={color}
                        strokeWidth={isSelected ? 3 : 2}
                      />
                      <text
                        x={node.x}
                        y={node.y + 4}
                        textAnchor="middle"
                        fill={isSelected ? 'white' : 'var(--color-text)'}
                        fontSize="12"
                        fontWeight={isSelected ? 'bold' : 'normal'}
                      >
                        {node.label.length > 12 ? node.label.slice(0, 11) + '…' : node.label}
                      </text>
                      <text
                        x={node.x}
                        y={node.y + radius + 14}
                        textAnchor="middle"
                        fill="var(--color-muted)"
                        fontSize="9"
                      >
                        {node.type}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>

          {selectedNode && (
            <div className="graph-detail-panel">
              <div className="graph-detail-header">
                <h3>{selectedNode.label}</h3>
                <span className="graph-detail-type" style={{ color: TYPE_COLORS[selectedNode.type] }}>
                  {selectedNode.type}
                </span>
                <button className="graph-detail-expand" onClick={() => handleNodeDoubleClick(selectedNode)}>
                  Expand
                </button>
              </div>
              {selectedNode.description && (
                <p className="graph-detail-desc">{selectedNode.description}</p>
              )}
              <div className="graph-detail-relations">
                <h4>Connected via</h4>
                <ul>
                  {filteredEdges.map((e, i) => (
                    <li key={i}>
                      <span className="relation-label">{e.label}</span>
                      <span className="relation-node">
                        {e.source === selectedNode.label ? e.target : e.source}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
              <button
                className="graph-detail-close"
                onClick={() => setSelectedNode(null)}
              >
                Close
              </button>
            </div>
          )}
        </div>
      </div>

      <style>{`
        .graph-overlay {
          position: fixed;
          inset: 0;
          z-index: 1000;
          background: rgba(0,0,0,0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
        }
        .graph-modal {
          background: var(--color-bg);
          border: 1px solid var(--color-border);
          border-radius: 16px;
          width: 100%;
          max-width: 1200px;
          height: 85vh;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .graph-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 24px;
          border-bottom: 1px solid var(--color-border);
        }
        .graph-header h2 { margin: 0; font-size: 18px; }
        .graph-header-actions { display: flex; align-items: center; gap: 12px; }
        .graph-stats-badge {
          font-size: 12px;
          color: var(--color-muted);
          background: var(--color-surface);
          padding: 4px 10px;
          border-radius: 20px;
        }
        .graph-close-btn {
          background: none;
          border: 1px solid var(--color-border);
          border-radius: 8px;
          padding: 6px;
          cursor: pointer;
          color: var(--color-text);
        }
        .graph-toolbar {
          display: flex;
          flex-direction: column;
          gap: 8px;
          padding: 12px 24px;
          border-bottom: 1px solid var(--color-border);
        }
        .graph-search { display: flex; gap: 8px; }
        .graph-search input {
          flex: 1;
          padding: 8px 12px;
          border-radius: 8px;
          border: 1px solid var(--color-border);
          background: var(--color-surface);
          color: var(--color-text);
          font-size: 14px;
        }
        .graph-search button {
          padding: 8px 16px;
          border-radius: 8px;
          border: none;
          background: var(--color-primary);
          color: white;
          cursor: pointer;
          font-weight: 500;
        }
        .graph-filters {
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
        }
        .graph-filter-btn {
          padding: 4px 12px;
          border-radius: 16px;
          border: 1px solid var(--color-border);
          background: transparent;
          color: var(--color-muted);
          cursor: pointer;
          font-size: 12px;
        }
        .graph-filter-btn.active {
          background: var(--color-primary);
          color: white;
          border-color: var(--color-primary);
        }
        .graph-content {
          flex: 1;
          display: flex;
          position: relative;
          overflow: hidden;
        }
        .graph-canvas-container {
          flex: 1;
          position: relative;
          overflow: hidden;
        }
        .graph-svg { display: block; }
        .graph-node:hover circle { filter: brightness(1.2); }
        .graph-loading, .graph-error, .graph-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          gap: 12px;
          color: var(--color-muted);
        }
        .graph-spinner {
          width: 32px;
          height: 32px;
          border: 3px solid var(--color-border);
          border-top-color: var(--color-primary);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .graph-error button {
          padding: 6px 16px;
          border-radius: 8px;
          border: none;
          background: var(--color-primary);
          color: white;
          cursor: pointer;
        }
        .graph-hint { font-size: 13px; }
        .graph-detail-panel {
          width: 300px;
          border-left: 1px solid var(--color-border);
          padding: 16px;
          overflow-y: auto;
          background: var(--color-surface);
        }
        .graph-detail-header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 8px; }
        .graph-detail-header h3 { margin: 0; font-size: 16px; }
        .graph-detail-type { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
        .graph-detail-expand {
          margin-left: auto;
          padding: 4px 10px;
          border-radius: 6px;
          border: 1px solid var(--color-border);
          background: transparent;
          color: var(--color-primary);
          cursor: pointer;
          font-size: 12px;
        }
        .graph-detail-desc { font-size: 13px; color: var(--color-muted); margin-bottom: 12px; line-height: 1.5; }
        .graph-detail-relations h4 { font-size: 13px; margin: 0 0 8px; color: var(--color-muted); }
        .graph-detail-relations ul {
          list-style: none;
          padding: 0;
          margin: 0;
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .graph-detail-relations li {
          display: flex;
          flex-direction: column;
          gap: 2px;
          padding: 8px;
          border-radius: 8px;
          background: var(--color-bg);
          font-size: 13px;
        }
        .relation-label { color: var(--color-primary); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
        .relation-node { color: var(--color-text); }
        .graph-detail-close {
          margin-top: 16px;
          width: 100%;
          padding: 8px;
          border-radius: 8px;
          border: 1px solid var(--color-border);
          background: transparent;
          color: var(--color-muted);
          cursor: pointer;
        }
      `}</style>
    </div>
  );
}
