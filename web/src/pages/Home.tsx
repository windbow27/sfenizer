import React from 'react';
import { Link } from 'react-router-dom';
import { Image, Video, Clock, ArrowRight } from 'lucide-react';
import { Card, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';

const features = [
  {
    to: '/image',
    icon: Image,
    title: 'Image Sfenizer',
    description: 'Upload or paste a shogi board photo to get SFEN/CSA notation.',
    cta: 'Convert Image'
  },
  {
    to: '/video',
    icon: Video,
    title: 'Video Sfenizer',
    description: 'Point your camera at a board and get real-time SFEN/CSA output.',
    cta: 'Open Camera'
  },
  {
    to: '/history',
    icon: Clock,
    title: 'History',
    description: 'Review and copy your previous image conversions.',
    cta: 'View History'
  }
];

const Home: React.FC = () => {
  return (
    <div className='py-8 md:py-12 max-w-3xl'>
      <div className='mb-10'>
        <div className='flex items-center gap-3 mb-4'>
          <img src='/piece/hitomoji_wood/black_pawn.png' alt='Logo' className='h-10' />
          <div>
            <h1 className='font-brand text-2xl sm:text-3xl font-bold'>Sfenizer</h1>
            <p className='text-sm text-muted-foreground'>
              Shogi board → SFEN/CSA notation converter
            </p>
          </div>
        </div>
      </div>

      <div className='space-y-3'>
        {features.map(({ to, icon: Icon, title, description, cta }) => (
          <Link key={to} to={to} className='block'>
            <Card>
              <CardContent className='p-5 flex items-center gap-4'>
                <Icon className='h-5 w-5 text-primary flex-shrink-0' />
                <div className='flex-1 min-w-0'>
                  <h2 className='text-sm font-semibold text-foreground'>{title}</h2>
                  <p className='text-sm text-muted-foreground'>{description}</p>
                </div>
                <span className='hidden sm:flex items-center gap-1 text-sm text-primary flex-shrink-0'>
                  {cta}
                  <ArrowRight className='h-4 w-4' />
                </span>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      <div className='mt-8 flex flex-wrap gap-3'>
        <Button asChild size='lg'>
          <Link to='/image'>
            <Image className='h-4 w-4 mr-2' />
            Convert an Image
          </Link>
        </Button>
        <Button asChild variant='outline' size='lg'>
          <Link to='/video'>
            <Video className='h-4 w-4 mr-2' />
            Start Live Camera
          </Link>
        </Button>
      </div>
    </div>
  );
};

export default Home;
