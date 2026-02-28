// ─── SFEN (Shogi Forsyth–Edwards Notation) parser & helpers ───

export type PieceColor = 'black' | 'white';
export type PieceKind =
  | 'king'
  | 'rook'
  | 'bishop'
  | 'gold'
  | 'silver'
  | 'knight'
  | 'lance'
  | 'pawn'
  | 'dragon'
  | 'horse'
  | 'prom_silver'
  | 'prom_knight'
  | 'prom_lance'
  | 'prom_pawn';

export interface Piece {
  color: PieceColor;
  kind: PieceKind;
}

export interface HandPieces {
  black: Partial<Record<HandPieceKind, number>>;
  white: Partial<Record<HandPieceKind, number>>;
}

/** Only unpromoted pieces (except king & gold) can appear in hand */
export type HandPieceKind = 'rook' | 'bishop' | 'gold' | 'silver' | 'knight' | 'lance' | 'pawn';

export interface ShogiPosition {
  board: (Piece | null)[][]; // 9×9, board[row][col], row 0 = rank 1 (top)
  turn: PieceColor;
  hands: HandPieces;
  moveNumber: number;
}

// ─── SFEN character → Piece mapping ───

const SFEN_PIECE_MAP: Record<string, Piece> = {
  K: { color: 'black', kind: 'king' },
  R: { color: 'black', kind: 'rook' },
  B: { color: 'black', kind: 'bishop' },
  G: { color: 'black', kind: 'gold' },
  S: { color: 'black', kind: 'silver' },
  N: { color: 'black', kind: 'knight' },
  L: { color: 'black', kind: 'lance' },
  P: { color: 'black', kind: 'pawn' },
  '+R': { color: 'black', kind: 'dragon' },
  '+B': { color: 'black', kind: 'horse' },
  '+S': { color: 'black', kind: 'prom_silver' },
  '+N': { color: 'black', kind: 'prom_knight' },
  '+L': { color: 'black', kind: 'prom_lance' },
  '+P': { color: 'black', kind: 'prom_pawn' },
  k: { color: 'white', kind: 'king' },
  r: { color: 'white', kind: 'rook' },
  b: { color: 'white', kind: 'bishop' },
  g: { color: 'white', kind: 'gold' },
  s: { color: 'white', kind: 'silver' },
  n: { color: 'white', kind: 'knight' },
  l: { color: 'white', kind: 'lance' },
  p: { color: 'white', kind: 'pawn' },
  '+r': { color: 'white', kind: 'dragon' },
  '+b': { color: 'white', kind: 'horse' },
  '+s': { color: 'white', kind: 'prom_silver' },
  '+n': { color: 'white', kind: 'prom_knight' },
  '+l': { color: 'white', kind: 'prom_lance' },
  '+p': { color: 'white', kind: 'prom_pawn' }
};

const HAND_CHAR_MAP: Record<string, { color: PieceColor; kind: HandPieceKind }> = {
  R: { color: 'black', kind: 'rook' },
  B: { color: 'black', kind: 'bishop' },
  G: { color: 'black', kind: 'gold' },
  S: { color: 'black', kind: 'silver' },
  N: { color: 'black', kind: 'knight' },
  L: { color: 'black', kind: 'lance' },
  P: { color: 'black', kind: 'pawn' },
  r: { color: 'white', kind: 'rook' },
  b: { color: 'white', kind: 'bishop' },
  g: { color: 'white', kind: 'gold' },
  s: { color: 'white', kind: 'silver' },
  n: { color: 'white', kind: 'knight' },
  l: { color: 'white', kind: 'lance' },
  p: { color: 'white', kind: 'pawn' }
};

// ─── Parse SFEN string ───

export function parseSfen(sfen: string): ShogiPosition {
  const parts = sfen.trim().split(/\s+/);
  const boardStr = parts[0] ?? '';
  const turnStr = parts[1] ?? 'b';
  const handsStr = parts[2] ?? '-';
  const moveStr = parts[3] ?? '1';

  // --- Board ---
  const board: (Piece | null)[][] = [];
  const ranks = boardStr.split('/');

  for (const rank of ranks) {
    const row: (Piece | null)[] = [];
    let i = 0;
    while (i < rank.length) {
      const ch = rank[i];
      if (ch >= '1' && ch <= '9') {
        const empties = parseInt(ch, 10);
        for (let e = 0; e < empties; e++) row.push(null);
        i++;
      } else if (ch === '+') {
        // Promoted piece: +X
        const token = rank.slice(i, i + 2);
        const piece = SFEN_PIECE_MAP[token];
        row.push(piece ? { ...piece } : null);
        i += 2;
      } else {
        const piece = SFEN_PIECE_MAP[ch];
        row.push(piece ? { ...piece } : null);
        i++;
      }
    }
    // Pad to 9 if short
    while (row.length < 9) row.push(null);
    board.push(row);
  }
  // Pad to 9 rows
  while (board.length < 9) {
    board.push(Array(9).fill(null));
  }

  // --- Turn ---
  const turn: PieceColor = turnStr === 'w' ? 'white' : 'black';

  // --- Hands ---
  const hands: HandPieces = { black: {}, white: {} };
  if (handsStr !== '-') {
    let j = 0;
    while (j < handsStr.length) {
      let count = 0;
      while (j < handsStr.length && handsStr[j] >= '0' && handsStr[j] <= '9') {
        count = count * 10 + parseInt(handsStr[j], 10);
        j++;
      }
      if (j < handsStr.length) {
        const entry = HAND_CHAR_MAP[handsStr[j]];
        if (entry) {
          if (count === 0) count = 1;
          hands[entry.color][entry.kind] = (hands[entry.color][entry.kind] ?? 0) + count;
        }
        j++;
      }
    }
  }

  return { board, turn, hands, moveNumber: parseInt(moveStr, 10) || 1 };
}

// ─── Piece → image path ───

const KIND_TO_FILE: Record<PieceKind, string> = {
  king: 'king',
  rook: 'rook',
  bishop: 'bishop',
  gold: 'gold',
  silver: 'silver',
  knight: 'knight',
  lance: 'lance',
  pawn: 'pawn',
  dragon: 'dragon',
  horse: 'horse',
  prom_silver: 'prom_silver',
  prom_knight: 'prom_knight',
  prom_lance: 'prom_lance',
  prom_pawn: 'prom_pawn'
};

export function pieceImagePath(piece: Piece): string {
  return `/piece/hitomoji_wood/${piece.color}_${KIND_TO_FILE[piece.kind]}.png`;
}

// ─── Board state → SFEN string (for round-trip) ───

const PIECE_TO_SFEN: Record<string, string> = {};
for (const [sfen, piece] of Object.entries(SFEN_PIECE_MAP)) {
  PIECE_TO_SFEN[`${piece.color}_${piece.kind}`] = sfen;
}

const HAND_KIND_ORDER: HandPieceKind[] = [
  'rook',
  'bishop',
  'gold',
  'silver',
  'knight',
  'lance',
  'pawn'
];
const HAND_SFEN_CHAR: Record<HandPieceKind, { black: string; white: string }> = {
  rook: { black: 'R', white: 'r' },
  bishop: { black: 'B', white: 'b' },
  gold: { black: 'G', white: 'g' },
  silver: { black: 'S', white: 's' },
  knight: { black: 'N', white: 'n' },
  lance: { black: 'L', white: 'l' },
  pawn: { black: 'P', white: 'p' }
};

export function positionToSfen(pos: ShogiPosition): string {
  // Board
  const ranks: string[] = [];
  for (const row of pos.board) {
    let rank = '';
    let empty = 0;
    for (const cell of row) {
      if (!cell) {
        empty++;
      } else {
        if (empty > 0) {
          rank += empty;
          empty = 0;
        }
        rank += PIECE_TO_SFEN[`${cell.color}_${cell.kind}`] ?? '?';
      }
    }
    if (empty > 0) rank += empty;
    ranks.push(rank);
  }

  // Hands
  let handStr = '';
  for (const color of ['black', 'white'] as const) {
    for (const kind of HAND_KIND_ORDER) {
      const count = pos.hands[color][kind];
      if (count && count > 0) {
        if (count > 1) handStr += count;
        handStr += HAND_SFEN_CHAR[kind][color];
      }
    }
  }
  if (!handStr) handStr = '-';

  return `${ranks.join('/')} ${pos.turn === 'white' ? 'w' : 'b'} ${handStr} ${pos.moveNumber}`;
}

/** Hand piece display order with Japanese labels */
export const HAND_PIECE_DISPLAY: { kind: HandPieceKind; kanji: string }[] = [
  { kind: 'rook', kanji: '飛' },
  { kind: 'bishop', kanji: '角' },
  { kind: 'gold', kanji: '金' },
  { kind: 'silver', kanji: '銀' },
  { kind: 'knight', kanji: '桂' },
  { kind: 'lance', kanji: '香' },
  { kind: 'pawn', kanji: '歩' }
];

/** Japanese rank labels */
export const RANK_KANJI = ['一', '二', '三', '四', '五', '六', '七', '八', '九'];
