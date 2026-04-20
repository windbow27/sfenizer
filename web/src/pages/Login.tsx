import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { KeyRound, LogIn, UserPlus } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { useAuth } from '../lib/auth';

const Login: React.FC = () => {
  const { login, register, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from || '/history';

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [isSubmitting, setIsSubmitting] = useState(false);

  React.useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [from, isAuthenticated, navigate]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsSubmitting(true);
    try {
      if (mode === 'login') {
        await login(username.trim(), password);
        toast.success('Logged in');
      } else {
        await register(username.trim(), password);
        toast.success('Account created');
      }
      navigate(from, { replace: true });
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Authentication failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className='mx-auto flex min-h-[calc(100vh-4rem)] items-center py-10 page-enter'>
      <div className='grid w-full gap-6 lg:grid-cols-[1.15fr_0.85fr]'>
        <div className='space-y-4 animate-fade-up'>
          <div className='inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground'>
            <KeyRound className='h-3.5 w-3.5 text-primary' />
            History
          </div>
          <div>
            <h1 className='text-3xl font-bold text-foreground'>Sign in to sync your conversions</h1>
            <p className='mt-3 max-w-md text-sm text-muted-foreground'>
              History is stored on the server, so login is what keeps your conversion list available
              across refreshes and devices.
            </p>
          </div>
        </div>

        <Card className='self-center animate-scale-in'>
          <CardContent className='p-6 space-y-4'>
            <div className='flex items-center gap-2'>
              <Button
                type='button'
                variant={mode === 'login' ? 'default' : 'outline'}
                size='sm'
                onClick={() => setMode('login')}
                className='flex-1 gap-2'>
                <LogIn className='h-4 w-4' />
                Log in
              </Button>
              <Button
                type='button'
                variant={mode === 'register' ? 'default' : 'outline'}
                size='sm'
                onClick={() => setMode('register')}
                className='flex-1 gap-2'>
                <UserPlus className='h-4 w-4' />
                Sign up
              </Button>
            </div>

            <form className='space-y-4' onSubmit={handleSubmit}>
              <div className='space-y-2'>
                <label className='text-sm font-medium text-foreground'>Username</label>
                <input
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  className='w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-primary'
                  autoComplete='username'
                  required
                  minLength={3}
                />
              </div>

              <div className='space-y-2'>
                <label className='text-sm font-medium text-foreground'>Password</label>
                <input
                  type='password'
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className='w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-primary'
                  autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                  required
                  minLength={6}
                />
              </div>

              <Button type='submit' className='w-full gap-2' disabled={isSubmitting}>
                {mode === 'login' ? (
                  <LogIn className='h-4 w-4' />
                ) : (
                  <UserPlus className='h-4 w-4' />
                )}
                {isSubmitting ? 'Please wait…' : mode === 'login' ? 'Log in' : 'Create account'}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Login;
