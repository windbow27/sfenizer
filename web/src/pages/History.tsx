import React, { useState, useEffect } from 'react';
import { Trash2, Copy, Check, Clock } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { toast } from 'sonner';
import { getHistory, removeFromHistory, clearHistory, type HistoryItem } from '../lib/history';

const RelativeTime: React.FC<{ timestamp: number }> = ({ timestamp }) => {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  let label: string;
  if (minutes < 1) label = 'just now';
  else if (minutes < 60) label = `${minutes}m ago`;
  else if (hours < 24) label = `${hours}h ago`;
  else label = `${days}d ago`;

  return (
    <span className='inline-flex items-center gap-1 text-xs text-muted-foreground'>
      <Clock className='h-3 w-3' />
      {label}
    </span>
  );
};

const HistoryCard: React.FC<{
  item: HistoryItem;
  onRemove: (id: string) => void;
}> = ({ item, onRemove }) => {
  const [copiedSfen, setCopiedSfen] = useState(false);
  const [copiedCsa, setCopiedCsa] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const copy = async (text: string, type: 'sfen' | 'csa') => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === 'sfen') {
        setCopiedSfen(true);
        setTimeout(() => setCopiedSfen(false), 2000);
      } else {
        setCopiedCsa(true);
        setTimeout(() => setCopiedCsa(false), 2000);
      }
      toast.success(`${type.toUpperCase()} copied!`);
    } catch {
      toast.error('Failed to copy');
    }
  };

  return (
    <Card className='transition-all duration-300 hover:shadow-md hover:-translate-y-0.5'>
      <CardContent className='p-4 space-y-3'>
        <div className='flex items-start gap-3'>
          {/* Thumbnail */}
          <img
            src={item.thumbnail}
            alt='Board'
            className='h-16 w-16 rounded-md object-cover flex-shrink-0 border border-border cursor-pointer'
            onClick={() => setExpanded((prev) => !prev)}
          />

          <div className='flex-1 min-w-0'>
            <div className='flex items-center justify-between gap-2'>
              <RelativeTime timestamp={item.timestamp} />
              <Button
                variant='ghost'
                size='sm'
                className='h-7 w-7 p-0 text-muted-foreground hover:text-destructive'
                onClick={() => onRemove(item.id)}>
                <Trash2 className='h-4 w-4' />
              </Button>
            </div>

            {/* SFEN row */}
            <div className='mt-1 flex items-center gap-2'>
              <p className='font-mono text-xs text-foreground truncate flex-1'>{item.sfen}</p>
              <Button
                variant='ghost'
                size='sm'
                className='h-6 w-6 p-0 flex-shrink-0'
                onClick={() => copy(item.sfen, 'sfen')}>
                {copiedSfen ? (
                  <Check className='h-3 w-3 text-green-500' />
                ) : (
                  <Copy className='h-3 w-3' />
                )}
              </Button>
            </div>

            <div className='flex items-center gap-2'>
              <p className='text-xs text-muted-foreground truncate flex-1'>CSA notation</p>
              <Button
                variant='ghost'
                size='sm'
                className='h-6 w-6 p-0 flex-shrink-0'
                onClick={() => copy(item.csa, 'csa')}>
                {copiedCsa ? (
                  <Check className='h-3 w-3 text-green-500' />
                ) : (
                  <Copy className='h-3 w-3' />
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Expanded image */}
        {expanded && (
          <img
            src={item.thumbnail}
            alt='Board large'
            className='w-full rounded-md object-contain max-h-72 border border-border cursor-pointer animate-scale-in'
            onClick={() => setExpanded(false)}
          />
        )}

        {/* Expanded CSA */}
        {expanded && (
          <div className='p-3 bg-muted rounded-md font-mono text-xs whitespace-pre-wrap animate-fade-up'>
            {item.csa}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

const History: React.FC = () => {
  const [items, setItems] = useState<HistoryItem[]>([]);

  useEffect(() => {
    setItems(getHistory());
  }, []);

  const handleRemove = (id: string) => {
    removeFromHistory(id);
    setItems((prev) => prev.filter((item) => item.id !== id));
    toast.success('Entry removed');
  };

  const handleClear = () => {
    clearHistory();
    setItems([]);
    toast.success('History cleared');
  };

  return (
    <div className='py-8 md:py-10 max-w-2xl page-enter'>
      <div className='space-y-6'>
        <div className='flex items-center justify-between animate-fade-up'>
          <div className='flex items-center gap-3'>
            <div className='h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center'>
              <Clock className='h-5 w-5 text-primary' />
            </div>
            <div>
              <h1 className='text-xl font-bold text-foreground'>History</h1>
              <p className='text-xs text-muted-foreground'>
                {items.length} conversion{items.length !== 1 ? 's' : ''} saved locally
              </p>
            </div>
          </div>
          {items.length > 0 && (
            <Button variant='outline' size='sm' onClick={handleClear} className='gap-2'>
              <Trash2 className='h-4 w-4' />
              Clear All
            </Button>
          )}
        </div>

        {items.length === 0 ? (
          <Card className='border-dashed animate-fade-up delay-100'>
            <CardContent className='p-0'>
              <div className='text-center py-16 text-muted-foreground'>
                <Clock className='h-12 w-12 mx-auto mb-3 opacity-20' />
                <p className='text-sm'>
                  No conversions yet — head to Image Sfenizer to get started.
                </p>
              </div>
            </CardContent>
          </Card>
        ) : (
          <div className='space-y-4'>
            {items.map((item, i) => (
              <div
                key={item.id}
                className='animate-fade-up'
                style={{ animationDelay: `${100 + i * 60}ms` }}>
                <HistoryCard item={item} onRemove={handleRemove} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default History;
