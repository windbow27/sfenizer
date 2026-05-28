import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Image, Video, Clock } from 'lucide-react';

const sidebarLinks = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/image', label: 'Image Sfenizer', icon: Image },
  { to: '/video', label: 'Video Sfenizer', icon: Video },
  { to: '/history', label: 'History', icon: Clock }
];

const Sidebar: React.FC = () => {
  const location = useLocation();

  return (
    <aside className='hidden md:flex flex-col w-52 flex-shrink-0 py-6'>
      <nav className='flex flex-col gap-0.5'>
        {sidebarLinks.map(({ to, label, icon: Icon }) => {
          const isActive = location.pathname === to;
          return (
            <Link
              key={to}
              to={to}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium ${
                isActive
                  ? 'text-primary bg-primary/5'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
              }`}>
              <Icon className='h-[18px] w-[18px]' />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
};

export default Sidebar;
