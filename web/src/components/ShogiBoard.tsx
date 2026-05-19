import React, { useMemo } from 'react';
import {
  parseSfen,
  pieceImagePath,
  HAND_PIECE_DISPLAY,
  RANK_KANJI,
  type Piece,
  type PieceColor,
  type HandPieceKind
} from '../lib/sfen';

interface ShogiBoardProps {
  /** SFEN string to render */
  sfen: string;
  /** Max width in px (default: 520) */
  maxWidth?: number;
}

const PieceImage: React.FC<{ piece: Piece }> = ({ piece }) => (
  <div className='w-full h-full flex items-center justify-center'>
    <img
      src={pieceImagePath(piece)}
      alt={`${piece.color} ${piece.kind}`}
      className='w-[88%] h-[88%] object-contain select-none pointer-events-none'
      draggable={false}
    />
  </div>
);

const HandSlot: React.FC<{
  color: PieceColor;
  kind: HandPieceKind;
  count: number;
}> = ({ color, kind, count }) => {
  if (count === 0) return null;
  return (
    <div
      className='relative flex items-center justify-center'
      style={{ width: '100%', aspectRatio: '1' }}>
      <img
        src={pieceImagePath({ color, kind })}
        alt={`${color} ${kind}`}
        className='w-[82%] h-[82%] object-contain select-none pointer-events-none'
        draggable={false}
      />
      {count > 1 && (
        <span className='absolute -bottom-0.5 -right-0.5 text-[10px] font-bold bg-primary text-primary-foreground rounded-full w-4 h-4 flex items-center justify-center leading-none shadow-sm'>
          {count}
        </span>
      )}
    </div>
  );
};

const ShogiBoard: React.FC<ShogiBoardProps> = ({ sfen, maxWidth = 520 }) => {
  const position = useMemo(() => parseSfen(sfen), [sfen]);

  const colLabels = [9, 8, 7, 6, 5, 4, 3, 2, 1];
  const handWidth = Math.round(maxWidth * 0.09);
  const labelWidth = Math.round(maxWidth * 0.035);

  return (
    <div
      className='flex items-stretch justify-center gap-1.5 sm:gap-2 select-none mx-auto'
      style={{ maxWidth }}>
      {/* ☖ Gote hand */}
      <div
        className='flex-shrink-0 rounded-lg border border-border/60 overflow-hidden shadow-sm flex flex-col'
        style={{
          width: handWidth,
          backgroundImage: 'url(/stand/wood_dark.png)',
          backgroundSize: 'cover'
        }}>
        <div className='text-center text-[10px] font-bold py-1 bg-black/30 text-white/90'>
          ☗ 後手
        </div>
        <div className='flex-1 flex flex-col items-center gap-0.5 p-1.5'>
          {HAND_PIECE_DISPLAY.map(({ kind }) => (
            <HandSlot
              key={kind}
              color='white'
              kind={kind}
              count={position.hands.white[kind] ?? 0}
            />
          ))}
        </div>
      </div>

      {/* Board */}
      <div className='flex-1 min-w-0'>
        {/* Column numbers */}
        <div className='flex' style={{ paddingLeft: labelWidth }}>
          {colLabels.map((n) => (
            <div
              key={n}
              className='flex-1 text-center text-[11px] font-semibold text-muted-foreground leading-tight pb-0.5'>
              {n}
            </div>
          ))}
        </div>

        <div className='flex'>
          {/* Row labels */}
          <div className='flex flex-col' style={{ width: labelWidth }}>
            {RANK_KANJI.map((k) => (
              <div
                key={k}
                className='flex-1 flex items-center justify-center text-[10px] text-muted-foreground'>
                {k}
              </div>
            ))}
          </div>

          {/* Grid */}
          <div
            className='relative flex-1 border-2 border-stone-800 rounded-sm overflow-hidden shadow-md'
            style={{
              aspectRatio: '1',
              backgroundImage: 'url(/board/wood_light.png)',
              backgroundSize: 'cover'
            }}>
            {/* SVG grid lines */}
            <img
              src='/board/grid_square.svg'
              alt=''
              className='absolute inset-0 w-full h-full pointer-events-none'
              draggable={false}
            />

            {/* Star points */}
            <div className='absolute inset-0 pointer-events-none'>
              {(
                [
                  [2, 6],
                  [6, 2],
                  [2, 2],
                  [6, 6]
                ] as const
              ).map(([r, c]) => (
                <div
                  key={`h${r}${c}`}
                  className='absolute w-1.5 h-1.5 bg-stone-800 rounded-full -translate-x-1/2 -translate-y-1/2'
                  style={{
                    left: `${((c + 1) / 9) * 100}%`,
                    top: `${((r + 1) / 9) * 100}%`
                  }}
                />
              ))}
            </div>

            {/* Pieces */}
            <div className='absolute inset-0 grid grid-cols-9 grid-rows-9'>
              {position.board.map((row, r) =>
                row.map((piece, c) => (
                  <div key={`${r}-${c}`} className='relative w-full h-full'>
                    {piece && <PieceImage piece={piece} />}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ☗ Sente hand */}
      <div
        className='flex-shrink-0 rounded-lg border border-border/60 overflow-hidden shadow-sm flex flex-col'
        style={{
          width: handWidth,
          backgroundImage: 'url(/stand/wood_dark.png)',
          backgroundSize: 'cover'
        }}>
        <div className='flex-1 flex flex-col-reverse items-center gap-0.5 p-1.5'>
          {HAND_PIECE_DISPLAY.map(({ kind }) => (
            <HandSlot
              key={kind}
              color='black'
              kind={kind}
              count={position.hands.black[kind] ?? 0}
            />
          ))}
        </div>
        <div className='text-center text-[10px] font-bold py-1 bg-black/30 text-white/90'>
          <span className='text-black'>☗ </span>先手
        </div>
      </div>
    </div>
  );
};

export default ShogiBoard;
