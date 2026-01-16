import type { Source, LogStep } from '../types';
import './SourcesDrawer.css';

interface SourcesDrawerProps {
  sources: Source[];
  logs: LogStep[];
  onClose: () => void;
}

export function SourcesDrawer({ sources, logs, onClose }: SourcesDrawerProps) {
  const totalTime = logs.reduce((sum, log) => sum + log.duration_ms, 0);

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <h2>Fontes e Logs</h2>
          <button className="drawer-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="drawer-content">
          <section className="drawer-section">
            <h3>📚 Fontes utilizadas ({sources.length})</h3>
            <div className="sources-list">
              {sources.map((source, index) => (
                <div key={index} className="source-card">
                  <div className="source-header">
                    <span className="source-rank">#{index + 1}</span>
                    <span className="source-similarity">
                      {(source.similarity * 100).toFixed(1)}% similar
                    </span>
                  </div>
                  <div className="source-info">
                    <div className="source-field">
                      <span className="field-label">Vídeo:</span>
                      <span className="field-value">{source.video_id}</span>
                    </div>
                    <div className="source-field">
                      <span className="field-label">Trecho:</span>
                      <span className="field-value">{source.chunk_id}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="drawer-section">
            <h3>⚙️ Logs de processamento</h3>
            <div className="logs-summary">
              <span>Tempo total: {totalTime}ms</span>
            </div>
            <div className="logs-list">
              {logs.map((log, index) => (
                <div key={index} className="log-item">
                  <div className="log-header">
                    <span className="log-step">{log.step}</span>
                    <span className="log-time">{log.duration_ms}ms</span>
                  </div>
                  {Object.keys(log.details).length > 0 && (
                    <pre className="log-details">
                      {JSON.stringify(log.details, null, 2)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
