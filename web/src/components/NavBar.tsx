import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Menu, X, Home, Image, Video, Clock, LogOut, LogIn } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from './ui/button';
import { Separator } from './ui/separator';
import { useAuth } from '../lib/auth';

const navLinks = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/image', label: 'Image', icon: Image },
  { to: '/video', label: 'Video', icon: Video },
  { to: '/history', label: 'History', icon: Clock }
];

const NavBar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const { user, isAuthenticated, isLoading, logout } = useAuth();

  const handleLogout = async () => {
    try {
      await logout();
      toast.success('Logged out');
      navigate('/');
    } catch {
      toast.error('Logout failed');
    }
  };

  return (
    <nav className='border-b border-border bg-card sticky top-0 z-50'>
      <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
        <div className='flex h-14 items-center justify-between'>
          <Link to='/' className='flex items-center gap-2'>
            <img src='/piece/hitomoji_wood/black_pawn.png' alt='Logo' className='h-8' />
            <span className='font-brand text-lg font-bold'>Sfenizer</span>
          </Link>

          <div className='hidden md:flex items-center gap-1'>
            {navLinks.map(({ to, label }) => (
              <Link
                key={to}
                to={to}
                className={`px-3 py-1.5 rounded-full text-sm font-medium ${
                  location.pathname === to
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}>
                {label}
              </Link>
            ))}
          </div>

          <div className='flex items-center gap-2'>
            {!isLoading && isAuthenticated ? (
              <div className='hidden sm:flex items-center gap-2'>
                <span className='text-xs text-muted-foreground'>Signed in as {user?.username}</span>
                <Button variant='outline' size='sm' onClick={handleLogout}>
                  <LogOut className='h-4 w-4 mr-2' />Logout
                </Button>
              </div>
            ) : (
              <Button variant='outline' size='sm' className='hidden sm:flex' asChild>
                <Link to='/login'><LogIn className='h-4 w-4 mr-2' />Login</Link>
              </Button>
            )}
            <Button variant='ghost' size='sm' className='md:hidden h-9 w-9 p-0' onClick={() => setMenuOpen((prev) => !prev)}>
              {menuOpen ? <X className='h-5 w-5' /> : <Menu className='h-5 w-5' />}
            </Button>
          </div>
        </div>
      </div>

      {menuOpen && (
        <div className='md:hidden border-t border-border bg-card'>
          <div className='px-4 py-2 space-y-0.5'>
            {navLinks.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                onClick={() => setMenuOpen(false)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium ${
                  location.pathname === to ? 'text-primary' : 'text-muted-foreground'
                }`}>
                <Icon className='h-4 w-4' />
                {label}
              </Link>
            ))}
            <Separator className='my-2' />
            {!isLoading && isAuthenticated ? (
              <Button variant='outline' size='sm' className='w-full justify-start' onClick={handleLogout}>
                <LogOut className='h-4 w-4 mr-2' />Logout{user?.username ? ` (${user.username})` : ''}
              </Button>
            ) : (
              <Button variant='outline' size='sm' className='w-full justify-start' asChild>
                <Link to='/login' onClick={() => setMenuOpen(false)}>
                  <LogIn className='h-4 w-4 mr-2' />Login
                </Link>
              </Button>
            )}
          </div>
        </div>
      )}
    </nav>
  );
};

export default NavBar;
