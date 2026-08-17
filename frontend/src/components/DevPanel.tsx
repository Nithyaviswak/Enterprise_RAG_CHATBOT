'use client';

import { useEffect, useState } from 'react';
import { DebugInfo, LiveMetrics } from '@/types';
import { getLiveMetrics } from '@/lib/api';

interface DevPanelProps {
  debug: DebugInfo | null;
}

export default function DevPanel({ debug }: DevPanelProps) {
  const [metrics, setMetrics] = useState<LiveMetrics | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  const refreshMetrics = async () => {
    setMetricsError(null);
    try {
      setMetrics(await getLiveMetrics());
    } catch (e: any) {
      setMetricsError(e.message || 'Failed to load metrics');
    }
  };

  useEffect(() => {
    refreshMetrics();
  }, []);

  const stages = debug?.stage_times ? Object.entries(debug.stage_times) : [];

  return (
    <div className="dev-panel">
      <div className="dev-panel-header">
        <span className="dev-panel-title">Developer Insights</span>
        <button className="dev-panel-refresh" onClick={refreshMetrics}>
          Refresh
        </button>
      </div>

      {/* Live runtime metrics */}
      <div className="dev-section">
        <h4>Live metrics (in-memory)</h4>
        {metricsError && <p className="dev-error">Error: {metricsError}</p>}
        {metrics && metrics.total_requests > 0 ? (
          <div className="dev-metric-grid">
            <div className="dev-metric">
              <div className="dev-metric-value">{metrics.total_requests}</div>
              <div className="dev-metric-label">requests</div>
            </div>
            <div className="dev-metric">
              <div className="dev-metric-value">
                {(metrics.avg_total_latency_ms ?? 0).toFixed(0)}ms
              </div>
              <div className="dev-metric-label">avg latency</div>
            </div>
            <div className="dev-metric">
              <div className="dev-metric-value">
                {(metrics.avg_retrieval_confidence ?? 0).toFixed(2)}
              </div>
              <div className="dev-metric-label">retrieval conf</div>
            </div>
            <div className="dev-metric">
              <div className="dev-metric-value">
                {(metrics.avg_grounding_ratio ?? 0).toFixed(2)}
              </div>
              <div className="dev-metric-label">grounding</div>
            </div>
            <div className="dev-metric">
              <div className="dev-metric-value">
                {(metrics.hallucination_risk_rate * 100).toFixed(1)}%
              </div>
              <div className="dev-metric-label">hallu risk</div>
            </div>
            <div className="dev-metric">
              <div className="dev-metric-value">
                {(metrics.refusal_rate * 100).toFixed(1)}%
              </div>
              <div className="dev-metric-label">refusal rate</div>
            </div>
          </div>
        ) : (
          <p className="dev-empty">
            No requests recorded yet — ask the assistant a question, then refresh.
          </p>
        )}
        {metrics && Object.keys(metrics.failure_counts || {}).length > 0 && (
          <div className="dev-failures">
            {Object.entries(metrics.failure_counts).map(([k, v]) => (
              <span key={k} className="meta-badge meta-badge-warn">
                {k}: {v}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Debug payload from last response */}
      <div className="dev-section">
        <h4>Last request</h4>
        {!debug ? (
          <p className="dev-empty">
            No debug payload yet — enable Dev Mode and ask a question.
          </p>
        ) : (
          <>
            <p className="dev-row">
              <span className="dev-label">request_id</span>
              <code>{debug.request_id}</code>
            </p>
            <p className="dev-row">
              <span className="dev-label">retrieval methods</span>
              <code>{debug.retrieval_methods.join(', ') || '—'}</code>
            </p>
            <p className="dev-row">
              <span className="dev-label">retrieval confidence</span>
              <code>{(debug.retrieval_confidence ?? 0).toFixed(3)}</code>
            </p>
            {stages.length > 0 && (
              <div className="dev-stages">
                {stages.map(([name, data]) => (
                  <div key={name} className="dev-stage">
                    <span className="dev-stage-name">{name}</span>
                    <span className={`dev-stage-status ${data.status}`}>{data.status}</span>
                    <span className="dev-stage-ms">{(data.latency_ms ?? 0).toFixed(0)}ms</span>
                  </div>
                ))}
              </div>
            )}
            {debug.retrieved_documents?.length > 0 && (
              <div className="dev-retrieved">
                {debug.retrieved_documents.map((doc, i) => (
                  <div key={i} className="dev-doc">
                    <div className="dev-doc-head">
                      <span>#{i + 1} {doc.source}{doc.page ? ` · p.${doc.page}` : ''}</span>
                      <span className="dev-doc-score">
                        {(doc.score ?? 0).toFixed(3)}
                        {doc.retrieval_method ? ` [${doc.retrieval_method}]` : ''}
                      </span>
                    </div>
                    <div className="dev-doc-excerpt">{(doc.excerpt || '').slice(0, 140)}…</div>
                  </div>
                ))}
              </div>
            )}
            {debug.system_prompt && (
              <details className="dev-details">
                <summary>system prompt</summary>
                <pre className="dev-pre">{debug.system_prompt}</pre>
              </details>
            )}
            {debug.final_context?.length > 0 && (
              <details className="dev-details">
                <summary>final context ({debug.final_context.length} chunks)</summary>
                {debug.final_context.map((c, i) => (
                  <pre key={i} className="dev-pre">{c.slice(0, 400)}</pre>
                ))}
              </details>
            )}
          </>
        )}
      </div>
    </div>
  );
}