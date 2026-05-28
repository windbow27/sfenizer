import React from 'react';
import { Link } from 'react-router-dom';
import { Image, Video, Clock } from 'lucide-react';
import { Card, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Separator } from '../components/ui/separator';

const features = [
  {
    to: '/image',
    icon: Image,
    title: 'Image Sfenizer',
    description: 'Upload or paste a shogi board photo to get SFEN/CSA notation.'
  },
  {
    to: '/video',
    icon: Video,
    title: 'Video Sfenizer',
    description: 'Point your camera at a board and get real-time SFEN/CSA output.'
  },
  {
    to: '/history',
    icon: Clock,
    title: 'History',
    description: 'Review and copy your previous image conversions.'
  }
];

const Home: React.FC = () => {
  return (
    <div className='py-8 max-w-3xl'>
      <div className='flex items-center gap-3 mb-6'>
        <img src='/piece/hitomoji_wood/black_pawn.png' alt='Logo' className='h-10' />
        <div>
          <h1 className='font-brand text-2xl font-bold'>Sfenizer</h1>
          <p className='text-sm text-muted-foreground'>Shogi board → SFEN/CSA notation converter</p>
        </div>
      </div>

      {features.map(({ to, icon: Icon, title, description }) => (
        <Link key={to} to={to}>
          <Card className='mb-4'>
            <CardContent className='p-4 flex items-center gap-3'>
              <Icon className='h-4 w-4 text-muted-foreground' />
              <div>
                <p className='text-sm font-medium'>{title}</p>
                <p className='text-xs text-muted-foreground'>{description}</p>
              </div>
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  );
};

export default Home;
