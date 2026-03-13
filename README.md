# MemeCV

Detecção de expressões faciais e gestos de mão em tempo real com exibição de memes correspondentes.

## Tecnologias

- **OpenCV** — captura e exibição de vídeo, desenho de landmarks
- **MediaPipe FaceMesh** — malha facial de 468 pontos
- **MediaPipe Hands** — rastreamento de até 2 mãos (21 landmarks cada)

## Gestos e expressões detectados

| Entrada | Meme exibido |
|---|---|
| Sorriso largo + boca aberta (FaceMesh) | `109fb257...jpg` — macaco sorrindo |
| Olhando para cima / pose pensativa (FaceMesh) | `maxresdefault.jpg` — macaco pensando |
| Sinal de paz ✌️ (Hands) | `109fb257...jpg` — macaco sorrindo |
| Joinha 👍 (Hands) | `7dc6efb0...jpg` — macaco joinha |
| Gesto de timeout T (duas mãos) | `bc3d38ff...jpg` — timeout |

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

Pressione **ESC** para encerrar.

> **Nota WSL2:** A webcam não é acessível diretamente no WSL2.
> Execute o programa a partir do terminal do Windows (cmd, PowerShell ou Windows Terminal).

## Estrutura do projeto

```
memecv/
├── main.py              # Código principal
├── requirements.txt     # Dependências Python
├── README.md            # Este arquivo
└── assets/
    └── new/
        ├── README.md    # Instruções sobre os memes
        ├── 109fb257daabe2f3db63bd7bc1944934.jpg   # Sorriso / paz
        ├── maxresdefault.jpg                       # Pensando (padrão)
        ├── 7dc6efb0fe7548ae00dd6143e739f630.jpg   # Joinha
        └── bc3d38ffc8a2e9a574bb54d3bffa5445.jpg   # Timeout
```

## Detalhes técnicos

### Estabilização de gestos

A detecção utiliza um histórico deslizante de **7 frames consecutivos** (`collections.deque`).
O meme só é trocado quando o mesmo gesto é confirmado em todos os 7 frames,
evitando mudanças bruscas por detecções espúrias.

### Landmarks utilizados (FaceMesh)

| Índice | Ponto |
|---|---|
| 13 | Lábio superior (centro) |
| 14 | Lábio inferior (centro) |
| 61 | Canto esquerdo da boca |
| 291 | Canto direito da boca |
| 1 | Ponta do nariz |
| 10 | Testa (glabela) |
| 152 | Queixo |

### Detecção do gesto de timeout (T)

O gesto é confirmado quando:
1. Duas mãos estão visíveis.
2. A ponta do indicador de uma mão está a menos de **0.18** (coordenadas normalizadas) do centro da palma da outra mão.
3. Pelo menos uma das mãos tem o indicador apontando para cima (ponta acima do pulso em mais de 0.05 unidades normalizadas) — formando o "cabo" do T.
