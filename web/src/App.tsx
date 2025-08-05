import React, { useState, useRef } from 'react';
import { Camera, Upload, Image, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

interface FileHandler {
  (file: File): void;
}

const App: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files[0]);
    }
  };

  const handleFiles: FileHandler = (file: File): void => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        if (e.target?.result) {
          setSelectedImage(e.target.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>): void => {
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files[0]);
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLDivElement>): void => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.startsWith('image/')) {
        const file = items[i].getAsFile();
        if (file) {
          handleFiles(file);
          break;
        }
      }
    }
  };

  const handleConvert = (): void => {
    if (selectedImage) {
      console.log('Converting image...');
      // Add your conversion logic here
    }
  };

  const clearImage = (): void => {
    setSelectedImage(null);
  };

  return (
    <div className='min-h-screen bg-background'>
      {/* Navbar */}
      <nav className='border-b border-border bg-card'>
        <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
          <div className='flex h-16 items-center justify-between'>
            <div className='flex items-center'>
              {/* <h1 className='text-xl font-semibold text-foreground'>VSC87</h1> */}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className='mx-auto max-w-2xl px-4 py-8 sm:px-6 lg:px-8'>
        <div className='space-y-6'>
          {/* Image Selector */}
          <Card className='overflow-hidden border-none shadow-none'>
            <CardContent className='p-0'>
              <div
                className={`relative rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
                  dragActive
                    ? 'border-primary bg-primary/5'
                    : selectedImage
                    ? 'border-primary bg-primary/5'
                    : 'border-border bg-card hover:border-primary/50 hover:bg-primary/5'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onPaste={handlePaste}
                tabIndex={0}>
                {selectedImage ? (
                  <div className='space-y-4'>
                    <div className='relative inline-block'>
                      <img
                        src={selectedImage}
                        alt='Selected'
                        className='mx-auto max-h-64 rounded-lg object-contain'
                      />
                      <Button
                        variant='secondary'
                        size='sm'
                        className='absolute -top-2 -right-2 h-6 w-6 rounded-full p-0'
                        onClick={clearImage}>
                        <X className='h-3 w-3' />
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className='space-y-4'>
                    <div className='mx-auto h-12 w-12 rounded-lg bg-muted flex items-center justify-center'>
                      <Image className='h-6 w-6 text-muted-foreground' />
                    </div>
                    <div className='space-y-2'>
                      <p className='text-lg font-medium text-foreground'>Drop your image here</p>
                      <p className='text-sm text-muted-foreground'>
                        Or use one of the options below
                      </p>
                    </div>

                    <div className='flex flex-wrap justify-center gap-3'>
                      <Button
                        variant='outline'
                        onClick={() => fileInputRef.current?.click()}
                        className='gap-2'>
                        <Upload className='h-4 w-4' />
                        Select Image
                      </Button>

                      <Button
                        variant='outline'
                        onClick={() => cameraInputRef.current?.click()}
                        className='gap-2'>
                        <Camera className='h-4 w-4' />
                        Camera
                      </Button>

                      <div className='inline-flex items-center gap-2 rounded-md border border-border bg-muted px-4 py-2 text-sm font-medium text-muted-foreground'>
                        <span>Ctrl+V</span>
                        <span className='text-xs'>Paste</span>
                      </div>
                    </div>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type='file'
                  accept='image/*'
                  onChange={handleFileSelect}
                  className='hidden'
                />
                <input
                  ref={cameraInputRef}
                  type='file'
                  accept='image/*'
                  capture='environment'
                  onChange={handleFileSelect}
                  className='hidden'
                />
              </div>
            </CardContent>
          </Card>

          <div className='flex justify-center'>
            <Button onClick={handleConvert} disabled={!selectedImage} size='lg' className='px-8'>
              Convert
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
