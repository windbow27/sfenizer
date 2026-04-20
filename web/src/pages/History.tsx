import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Trash2, Copy, Check, Clock, LogIn } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { toast } from 'sonner';
import { getHistory, removeFromHistory, clearHistory, type HistoryItem } from '../lib/history';
import { useAuth } from '../lib/auth';
import { formatRelativeTime } from '../lib/dates';

const RelativeTime: React.FC<{ timestamp: number }> = ({ timestamp }) => {
  return (
    <span className='inline-flex items-center gap-1 text-xs text-muted-foreground'>
      <Clock className='h-3 w-3' />
      {formatRelativeTime(timestamp)}
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
        if (active) {
          setItems(historyItems);
          setLoadError(null);
        }
      })
      .catch((error) => {
        if (active) {
          setLoadError(error instanceof Error ? error.message : 'Failed to load history');
        }
      })
      .finally(() => {
        if (active) {
          setIsLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [isAuthenticated]);

  if (isAuthLoading || isLoading) {
    return (
      <div className='py-8 md:py-10 max-w-2xl page-enter'>
        <Card className='animate-pulse'>
          <CardContent className='p-6'>
            <div className='h-5 w-32 rounded bg-muted mb-4' />
            <div className='h-24 rounded bg-muted' />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className='py-8 md:py-10 max-w-2xl page-enter'>
        <Card className='border-dashed animate-fade-up'>
          <CardContent className='p-6 text-center space-y-4'>
            <Clock className='h-12 w-12 mx-auto opacity-20' />
            <div className='space-y-1'>
              <h1 className='text-xl font-bold text-foreground'>History is saved on the server</h1>
              <p className='text-sm text-muted-foreground'>
                Sign in to see your conversions and keep them across sessions.
              </p>
            </div>
            <Button asChild className='gap-2'>
              <Link to='/login'>
                <LogIn className='h-4 w-4' />
                Log in
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className='py-8 md:py-10 max-w-2xl page-enter'>
        <Card className='border-dashed animate-fade-up'>
          <CardContent className='p-6 text-center space-y-4'>
            <Clock className='h-12 w-12 mx-auto opacity-20' />
            <p className='text-sm text-muted-foreground'>{loadError}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const handleRemove = (id: string) => {
    removeFromHistory(id)
      .then(() => {
        setItems((prev) => prev.filter((item) => item.id !== id));
        toast.success('Entry removed');
      })
      .catch(() => toast.error('Failed to remove entry'));
  };

  const handleClear = () => {
    clearHistory()
      .then(() => {
        setItems([]);
        toast.success('History cleared');
      })
      .catch(() => toast.error('Failed to clear history'));
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
                {items.length} conversion{items.length !== 1 ? 's' : ''} saved on the server
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
