import { useEffect, useRef, useState } from 'react';
import { useChat } from '../hooks/useChat';
import { Message } from './Message';
import { InputBar } from './InputBar';
import { Suggestions } from './Suggestions';
import { TypingIndicator } from './TypingIndicator';
import './Chat.css';

export function Chat() {
  const { messages, isLoading, sendMessage } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showSuggestions, setShowSuggestions] = useState(true);

  // Auto-scroll para última mensagem
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Esconde sugestões após primeira mensagem do usuário
  useEffect(() => {
    const hasUserMessage = messages.some((m) => m.role === 'user');
    if (hasUserMessage) {
      setShowSuggestions(false);
    }
  }, [messages]);

  const handleSend = (text: string) => {
    sendMessage(text);
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="header-info">
          <div className="header-avatar">📚</div>
          <div className="header-text">
            <h1>SOS Educação</h1>
            <span className="header-status">
              {isLoading ? 'Digitando...' : 'Assistente de Alfabetização'}
            </span>
          </div>
        </div>
      </header>

      <main className="chat-messages">
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        
        {isLoading && <TypingIndicator />}
        
        <div ref={messagesEndRef} />
      </main>

      {showSuggestions && (
        <Suggestions onSelect={handleSend} disabled={isLoading} />
      )}

      <InputBar onSend={handleSend} disabled={isLoading} />
    </div>
  );
}
