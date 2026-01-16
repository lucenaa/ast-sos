import { useState } from 'react';
import type { Message as MessageType } from '../types';
import { SourcesDrawer } from './SourcesDrawer';
import './Message.css';

interface MessageProps {
  message: MessageType;
}

export function Message({ message }: MessageProps) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';
  const hasSources = message.sources && message.sources.length > 0;

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('pt-BR', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Parse markdown-like formatting
  const formatContent = (content: string) => {
    // Bold
    let formatted = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Line breaks
    formatted = formatted.replace(/\n/g, '<br />');
    return formatted;
  };

  return (
    <>
      <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
        {!isUser && <div className="message-avatar">🤖</div>}
        
        <div className="message-content-wrapper">
          <div
            className={`message-bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}
            dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
          />
          
          <div className="message-meta">
            <span className="message-time">{formatTime(message.timestamp)}</span>
            {hasSources && (
              <button
                className="sources-toggle"
                onClick={() => setShowSources(true)}
              >
                📚 Ver fontes ({message.sources!.length})
              </button>
            )}
          </div>
        </div>
        
        {isUser && <div className="message-avatar">👤</div>}
      </div>

      {showSources && message.sources && message.logs && (
        <SourcesDrawer
          sources={message.sources}
          logs={message.logs}
          onClose={() => setShowSources(false)}
        />
      )}
    </>
  );
}
