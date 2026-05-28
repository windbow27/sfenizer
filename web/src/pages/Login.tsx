import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { LogIn, UserPlus } from 'lucide-react';
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
    <div className='mx-auto flex min-h-[calc(100vh-4rem)] items-center py-10'>
      <div className='grid w-full gap-6 lg:grid-cols-2'>
        <div className='space-y-3'>
          <h1 className='text-3xl font-bold'>Sign in to sync your conversions</h1>
          <p className='text-sm text-muted-foreground max-w-md'>
            History is stored on the server, so login is what keeps your conversion list available
            across refreshes and devices.
          </p>
        </div>

        <Card className='self-center'>
          <CardContent className='p-6 space-y-4'>
            <div className='flex gap-2'>
              <Button
                type='button'
                variant={mode === 'login' ? 'default' : 'outline'}
                size='sm'
                onClick={() => setMode('login')}
                className='flex-1'>
                <LogIn className='h-4 w-4 mr-2' />
                Log in
              </Button>
              <Button
                type='button'
                variant={mode === 'register' ? 'default' : 'outline'}
                size='sm'
                onClick={() => setMode('register')}
                className='flex-1'>
                <UserPlus className='h-4 w-4 mr-2' />
                Sign up
              </Button>
            </div>

            <form className='space-y-4' onSubmit={handleSubmit}>
              <div className='space-y-1'>
                <label className='text-sm font-medium'>Username</label>
                <input
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  className='w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary'
                  autoComplete='username'
                  required
                  minLength={3}
                />
              </div>

              <div className='space-y-1'>
                <label className='text-sm font-medium'>Password</label>
                <input
                  type='password'
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className='w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary'
                  autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                  required
                  minLength={6}
                />
              </div>

              <Button type='submit' className='w-full' disabled={isSubmitting}>
                {mode === 'login' ? <LogIn className='h-4 w-4 mr-2' /> : <UserPlus className='h-4 w-4 mr-2' />}
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
