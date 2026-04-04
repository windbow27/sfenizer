import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Home, Image, Video, Clock } from 'lucide-react';
import { Button } from './ui/button';

const navLinks = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/image', label: 'Image', icon: Image },
  { to: '/video', label: 'Video', icon: Video },
  { to: '/history', label: 'History', icon: Clock }
];

const NavBar: React.FC = () => {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className='border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-50 animate-fade-down'>
      <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
        <div className='flex h-14 items-center justify-between'>
          {/* Brand */}
          <Link to='/' className='flex items-center gap-2.5 group'>
            <div className='flex items-center justify-center transition-transform duration-300 group-hover:scale-110 group-hover:rotate-3'>
              <img src='/piece/hitomoji_wood/black_pawn.png' alt='Logo' className='h-8' />
            </div>
            <span className='font-brand text-lg font-bold text-foreground tracking-tight'>
              Sfenizer
            </span>
          </Link>

          {/* Desktop nav pills */}
          <div className='hidden md:flex items-center gap-1'>
            {navLinks.map((link, i) => {
              const isActive = location.pathname === link.to;
              return (
                <Link
                  key={link.to}
                  to={link.to}
                  className={`px-3.5 py-1.5 rounded-full text-sm font-medium transition-all duration-200 animate-fade-in ${
                    isActive
                      ? 'bg-primary text-primary-foreground shadow-sm'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted/60 hover:scale-105 active:scale-95'
                  }`}
                  style={{ animationDelay: `${100 + i * 50}ms` }}>
                  {link.label}
                </Link>
              );
            })}
          </div>

          {/* Right side placeholder + mobile hamburger */}
          <div className='flex items-center gap-2'>
            {/* <span className='hidden sm:block text-xs text-muted-foreground'>v2</span> */}
            <Button
              variant='ghost'
              size='sm'
              className='md:hidden h-9 w-9 p-0'
              onClick={() => setMenuOpen((prev) => !prev)}>
              {menuOpen ? <X className='h-5 w-5' /> : <Menu className='h-5 w-5' />}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className='md:hidden border-t border-border bg-card animate-fade-down overflow-hidden'>
          <div className='px-4 py-2 space-y-0.5'>
            {navLinks.map(({ to, label, icon: Icon }, i) => {
              const isActive = location.pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  onClick={() => setMenuOpen(false)}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 animate-fade-up active:scale-[0.98] ${
                    isActive
                      ? 'text-primary bg-primary/5'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                  }`}
                  style={{ animationDelay: `${i * 50}ms` }}>
                  <Icon className='h-[18px] w-[18px]' />
                  {label}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </nav>
  );
};

export default NavBar;
