import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Trash2, Copy, Check, Clock, LogIn } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { Separator } from '../components/ui/separator';
import { toast } from 'sonner';
import { getHistory, removeFromHistory, clearHistory, type HistoryItem } from '../lib/history';
import { useAuth } from '../lib/auth';
import { formatRelativeTime } from '../lib/dates';

const HistoryCard: React.FC<{ item: HistoryItem; onRemove: (id: string) => void }> = ({
  item,
  onRemove
}) => {
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
    <Card>
      <CardContent className='p-4 space-y-3'>
        <div className='flex items-start gap-3'>
          <img
            src={item.thumbnail}
            alt='Board'
            className='h-16 w-16 rounded-md object-cover border border-border cursor-pointer'
            onClick={() => setExpanded((prev) => !prev)}
          />
          <div className='flex-1 min-w-0'>
            <div className='flex items-center justify-between'>
              <span className='flex items-center gap-1 text-xs text-muted-foreground'>
                <Clock className='h-3 w-3' />
                {formatRelativeTime(item.timestamp)}
              </span>
              <Button variant='ghost' size='sm' className='h-7 w-7 p-0' onClick={() => onRemove(item.id)}>
                <Trash2 className='h-4 w-4' />
              </Button>
            </div>

            <div className='flex items-center gap-2 mt-1'>
              <p className='font-mono text-xs truncate flex-1'>{item.sfen}</p>
              <Button variant='ghost' size='sm' className='h-6 w-6 p-0' onClick={() => copy(item.sfen, 'sfen')}>
                {copiedSfen ? <Check className='h-3 w-3 text-green-500' /> : <Copy className='h-3 w-3' />}
              </Button>
            </div>

            <div className='flex items-center gap-2'>
              <p className='text-xs text-muted-foreground truncate flex-1'>CSA notation</p>
              <Button variant='ghost' size='sm' className='h-6 w-6 p-0' onClick={() => copy(item.csa, 'csa')}>
                {copiedCsa ? <Check className='h-3 w-3 text-green-500' /> : <Copy className='h-3 w-3' />}
              </Button>
            </div>
          </div>
        </div>

        {expanded && (
          <>
            <img
              src={item.thumbnail}
              alt='Board large'
              className='w-full rounded-md object-contain max-h-72 border border-border cursor-pointer'
              onClick={() => setExpanded(false)}
            />
            <pre className='p-3 bg-muted rounded-md text-xs whitespace-pre-wrap font-mono'>{item.csa}</pre>
          </>
        )}
      </CardContent>
    </Card>
  );
};

const History: React.FC = () => {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const { isAuthenticated, isLoading: isAuthLoading } = useAuth();

  useEffect(() => {
    if (!isAuthenticated) {
      setItems([]);
      setLoadError(null);
      setIsLoading(false);
      return;
    }

    let active = true;
    setIsLoading(true);
    getHistory()
      .then((historyItems) => {
        if (active) { setItems(historyItems); setLoadError(null); }
      })
      .catch((error) => {
        if (active) setLoadError(error instanceof Error ? error.message : 'Failed to load history');
      })
      .finally(() => { if (active) setIsLoading(false); });

    return () => { active = false; };
  }, [isAuthenticated]);

  if (isAuthLoading || isLoading) {
    return (
      <div className='py-8 max-w-2xl'>
        <Card><CardContent className='p-6'><p className='text-sm text-muted-foreground'>Loading…</p></CardContent></Card>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className='py-8 max-w-2xl'>
        <Card>
          <CardContent className='p-6 text-center space-y-4'>
            <p className='font-semibold'>History is saved on the server</p>
            <p className='text-sm text-muted-foreground'>Sign in to see your conversions.</p>
            <Button asChild>
              <Link to='/login'><LogIn className='h-4 w-4 mr-2' />Log in</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className='py-8 max-w-2xl'>
        <Card><CardContent className='p-6'><p className='text-sm text-muted-foreground'>{loadError}</p></CardContent></Card>
      </div>
    );
  }

  const handleRemove = (id: string) => {
    removeFromHistory(id)
      .then(() => { setItems((prev) => prev.filter((item) => item.id !== id)); toast.success('Entry removed'); })
      .catch(() => toast.error('Failed to remove entry'));
  };

  const handleClear = () => {
    clearHistory()
      .then(() => { setItems([]); toast.success('History cleared'); })
      .catch(() => toast.error('Failed to clear history'));
  };

  return (
    <div className='py-8 max-w-2xl'>
      <div className='space-y-4'>
        <div className='flex items-center justify-between'>
          <div>
            <h1 className='text-xl font-bold'>History</h1>
            <p className='text-xs text-muted-foreground'>
              {items.length} conversion{items.length !== 1 ? 's' : ''} on server
            </p>
          </div>
          {items.length > 0 && (
            <Button variant='outline' size='sm' onClick={handleClear}>
              <Trash2 className='h-4 w-4 mr-2' />Clear All
            </Button>
          )}
        </div>

        <Separator />

        {items.length === 0 ? (
          <p className='text-sm text-muted-foreground py-8 text-center'>
            No conversions yet — head to Image Sfenizer to get started.
          </p>
        ) : (
          <div className='space-y-3'>
            {items.map((item) => (
              <HistoryCard key={item.id} item={item} onRemove={handleRemove} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default History;
