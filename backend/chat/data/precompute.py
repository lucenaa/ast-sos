"""
Script para pré-computar embeddings do CSV para arquivo .npy.

Uso:
    python precompute.py

Isso converte a coluna 'embedding' do CSV para uma matriz numpy,
permitindo carregamento instantâneo no startup do servidor.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_embedding(cell) -> list[float] | None:
    """Converte célula do CSV para lista de floats."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    if isinstance(cell, list):
        try:
            return [float(x) for x in cell]
        except Exception:
            return None
    s = str(cell).strip()
    if not s:
        return None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        return None
    return None


def main():
    data_dir = Path(__file__).parent
    csv_path = data_dir / "documents_rows.csv"
    npy_path = data_dir / "embeddings.npy"
    
    if not csv_path.exists():
        print(f"Erro: CSV não encontrado em {csv_path}")
        sys.exit(1)
    
    print(f"Carregando CSV de {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    print(f"Linhas carregadas: {len(df)}")
    
    # Encontra coluna de embedding
    cols_lower = {c.lower(): c for c in df.columns}
    embedding_col = None
    for key in ["embedding", "embeddings", "vector"]:
        if key in cols_lower:
            embedding_col = cols_lower[key]
            break
    
    if embedding_col is None:
        print("Erro: Coluna de embedding não encontrada")
        sys.exit(1)
    
    print(f"Parseando coluna '{embedding_col}'...")
    embeddings = []
    valid_count = 0
    invalid_count = 0
    
    for i, cell in enumerate(df[embedding_col].tolist()):
        emb = parse_embedding(cell)
        if emb is not None:
            embeddings.append(emb)
            valid_count += 1
        else:
            embeddings.append([0.0])  # placeholder
            invalid_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processados: {i + 1}/{len(df)}")
    
    print(f"Válidos: {valid_count}, Inválidos: {invalid_count}")
    
    # Filtra apenas os válidos e cria matriz
    valid_embeddings = [e for e in embeddings if len(e) > 1]
    
    if not valid_embeddings:
        print("Erro: Nenhum embedding válido encontrado")
        sys.exit(1)
    
    print(f"Criando matriz numpy com {len(valid_embeddings)} embeddings...")
    matrix = np.array(valid_embeddings, dtype=np.float32)
    
    # Normaliza
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    
    print(f"Shape da matriz: {matrix.shape}")
    
    # Salva
    np.save(npy_path, matrix)
    print(f"Matriz salva em {npy_path}")
    print(f"Tamanho do arquivo: {npy_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
