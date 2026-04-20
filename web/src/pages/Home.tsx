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
    description: 'Upload or paste a shogi board photo to get SFEN/CSA notation instantly.',
    cta: 'Convert Image',
    badge: 'sfenizer.pth'
  },
  {
    to: '/video',
    icon: Video,
    title: 'Video Sfenizer',
    description: 'Point your camera at a board and get real-time SFEN/CSA output via YOLO.',
    cta: 'Open Camera',
    badge: 'yolo11m.pt'
  },
  {
    to: '/history',
    icon: Clock,
    title: 'History',
    description: 'Review and copy your previous image conversions from the server.',
    cta: 'View History',
    badge: 'SQLite'
  }
];

const Home: React.FC = () => {
  return (
    <div className='py-8 md:py-12 max-w-3xl page-enter'>
      {/* Welcome section */}
      <div className='mb-10 animate-fade-up'>
        <div className='flex items-center gap-3 mb-4'>
          <div className='flex items-center justify-center animate-bounce-subtle'>
            <img src='/piece/hitomoji_wood/black_pawn.png' alt='Logo' className='h-10' />
          </div>
          <div>
            <h1 className='font-brand text-2xl sm:text-3xl font-bold text-foreground'>Sfenizer</h1>
            <p className='text-sm text-muted-foreground'>
              Shogi board → SFEN/CSA notation converter
            </p>
          </div>
        </div>
        {/* <div className='flex items-center gap-2 text-xs text-muted-foreground animate-fade-in delay-300'>
          <Sparkles className='h-3.5 w-3.5 text-primary' />
          <span>Powered by ResNet-18 &amp; YOLOv11</span>
        </div> */}
      </div>

      {/* Feature cards */}
      <div className='space-y-4'>
        {features.map(({ to, icon: Icon, title, description, cta, badge }, i) => (
          <Link key={to} to={to} className='block group'>
            <Card
              className='group-hover:border-primary/30 transition-all duration-300 group-hover:shadow-md group-hover:-translate-y-0.5 animate-fade-up'
              style={{ animationDelay: `${150 + i * 100}ms` }}>
              <CardContent className='p-5 sm:p-6 flex items-center gap-5'>
                <div className='h-12 w-12 rounded-xl bg-primary/8 flex items-center justify-center flex-shrink-0 group-hover:bg-primary/15 transition-all duration-300 group-hover:scale-110 group-hover:rotate-3'>
                  <Icon className='h-6 w-6 text-primary' />
                </div>
                <div className='flex-1 min-w-0'>
                  <div className='flex items-center gap-2'>
                    <h2 className='text-base font-semibold text-foreground'>{title}</h2>
                    <span className='text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground hidden sm:inline-block'>
                      {badge}
                    </span>
                  </div>
                  <p className='text-sm text-muted-foreground mt-0.5'>{description}</p>
                </div>
                <div className='hidden sm:flex items-center gap-1 text-sm font-medium text-primary opacity-0 group-hover:opacity-100 transition-all duration-300 group-hover:translate-x-1 flex-shrink-0'>
                  {cta}
                  <ArrowRight className='h-4 w-4' />
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {/* Quick actions */}
      <div className='mt-8 flex flex-wrap gap-3 animate-fade-up delay-500'>
        <Button
          asChild
          size='lg'
          className='gap-2 transition-transform duration-200 hover:scale-105 active:scale-95'>
          <Link to='/image'>
            <Image className='h-4 w-4' />
            Convert an Image
          </Link>
        </Button>
        <Button
          asChild
          variant='outline'
          size='lg'
          className='gap-2 transition-transform duration-200 hover:scale-105 active:scale-95'>
          <Link to='/video'>
            <Video className='h-4 w-4' />
            Start Live Camera
          </Link>
        </Button>
      </div>
    </div>
  );
};

export default Home;
