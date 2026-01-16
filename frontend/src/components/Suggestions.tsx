import './Suggestions.css';

interface SuggestionsProps {
  onSelect: (text: string) => void;
  disabled?: boolean;
}

const SUGGESTIONS = [
  'Como incentivar o gosto pela leitura?',
  'Ideias de atividades de consciência fonológica',
  'Sequência didática para o 1º ano',
  'Como envolver as famílias na alfabetização?',
  'Estratégias para alunos com dificuldade',
];

export function Suggestions({ onSelect, disabled }: SuggestionsProps) {
  return (
    <div className="suggestions-container">
      <p className="suggestions-label">Sugestões de perguntas:</p>
      <div className="suggestions-grid">
        {SUGGESTIONS.map((text, index) => (
          <button
            key={index}
            className="suggestion-chip"
            onClick={() => onSelect(text)}
            disabled={disabled}
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}
