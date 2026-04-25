import React, { useState, useRef } from 'react';
import { Camera, Upload, Image, X, Check, Copy } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { toast } from 'sonner';
import { addToHistory } from '../lib/history';
import { useAuth } from '../lib/auth';
import ShogiBoard from '../components/ShogiBoard';

interface ConversionResult {
  success: boolean;
  sfen: string;
  csa: string;
  board: string[][];
}

const API_BASE_URL = import.meta.env.PROD ? '/api' : 'http://localhost:8000';

const ImageSfenizer: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ConversionResult | null>(null);
  const [copiedSfen, setCopiedSfen] = useState(false);
  const [copiedCsa, setCopiedCsa] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const copyToClipboard = async (text: string, type: 'sfen' | 'csa') => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === 'sfen') {
        setCopiedSfen(true);
        setTimeout(() => setCopiedSfen(false), 2000);
      } else {
        setCopiedCsa(true);
        setTimeout(() => setCopiedCsa(false), 2000);
      }
      toast.success(`${type.toUpperCase()} copied to clipboard!`);
    } catch {
      toast.error('Failed to copy to clipboard');
    }
  };

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFiles(e.dataTransfer.files[0]);
  };

  const handleFiles = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) setSelectedImage(e.target.result as string);
    };
    reader.readAsDataURL(file);
    setResult(null);
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLDivElement>) => {
    for (const item of Array.from(e.clipboardData.items)) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          handleFiles(file);
          break;
        }
      }
    }
  };

  const handleConvert = async () => {
    if (!selectedFile || !selectedImage) {
      toast.error('No image selected');
      return;
    }
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${API_BASE_URL}/convert`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Conversion failed');
      }

      const data: ConversionResult = await response.json();
      setResult(data);
      toast.success('Board converted successfully!');

      if (isAuthenticated) {
        try {
          await addToHistory({
            timestamp: Date.now(),
            thumbnail: selectedImage,
            sfen: data.sfen,
            csa: data.csa,
            board: data.board
          });
          toast.success('Saved to server history');
        } catch (historyError) {
          toast.error(
            historyError instanceof Error ? historyError.message : 'Failed to save history'
          );
        }
      } else {
        toast.info('Log in to save conversions to history');
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setResult(null);
  };

  return (
    <div className='py-8 md:py-10 max-w-3xl page-enter'>
      <div className='space-y-6'>
        <div className='flex items-center gap-3 animate-fade-up'>
          <div className='h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center'>
            <Image className='h-5 w-5 text-primary' />
          </div>
          <div>
            <h1 className='text-xl font-bold text-foreground'>Image Sfenizer</h1>
            <p className='text-xs text-muted-foreground'>
              Upload a shogi board photo to extract SFEN/CSA notation
            </p>
          </div>
        </div>

        {/* Drop zone */}
        <Card className='overflow-hidden border-none shadow-none animate-fade-up delay-100'>
          <CardContent className='p-0'>
            <div
              className={`relative rounded-lg border-2 border-dashed p-8 text-center transition-all duration-300 outline-none ${
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
                      className='mx-auto max-h-64 rounded-lg object-contain animate-scale-in'
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
                  <div className='space-y-1'>
                    <p className='text-lg font-medium text-foreground'>
                      Drop your shogi board image here
                    </p>
                    <p className='text-sm text-muted-foreground'>
                      Or use one of the options below — Ctrl+V also works
                    </p>
                  </div>
                  <div className='flex flex-wrap justify-center gap-3'>
                    <Button
                      variant='outline'
                      onClick={() => fileInputRef.current?.click()}
                      className='gap-2'>
                      <Upload className='h-4 w-4' />
                      Select File
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
                onChange={(e) => e.target.files?.[0] && handleFiles(e.target.files[0])}
                className='hidden'
              />
              <input
                ref={cameraInputRef}
                type='file'
                accept='image/*'
                capture='environment'
                onChange={(e) => e.target.files?.[0] && handleFiles(e.target.files[0])}
                className='hidden'
              />
            </div>
          </CardContent>
        </Card>

        <div className='flex justify-center animate-fade-up delay-200'>
          <Button
            onClick={handleConvert}
            disabled={!selectedFile || isLoading}
            size='lg'
            className='px-8 transition-transform duration-200 hover:scale-105 active:scale-95'>
            {isLoading ? 'Converting…' : 'Convert to SFEN/CSA'}
          </Button>
        </div>

        {/* Results */}
        {result && (
          <div className='space-y-6 animate-scale-in'>
            {/* Board visualization */}
            <div>
              <h3 className='text-base font-semibold flex items-center gap-2 mb-4'>
                <Check className='h-4 w-4 text-primary' />
                Detected Board
              </h3>
              <ShogiBoard sfen={result.sfen} maxWidth={520} />
            </div>

            {/* Notation */}
            <Card className='border-primary/20'>
              <CardContent className='p-5 sm:p-6 space-y-4'>
                <div>
                  <div className='flex items-center justify-between mb-2'>
                    <label className='text-sm font-medium text-muted-foreground'>SFEN:</label>
                    <Button
                      variant='ghost'
                      size='sm'
                      onClick={() => copyToClipboard(result.sfen, 'sfen')}
                      className='h-8 w-8 p-0'>
                      {copiedSfen ? (
                        <Check className='h-4 w-4 text-green-500' />
                      ) : (
                        <Copy className='h-4 w-4' />
                      )}
                    </Button>
                  </div>
                  <div className='p-3 bg-muted rounded-md font-mono text-sm break-all'>
                    {result.sfen}
                  </div>
                </div>

                <div>
                  <div className='flex items-center justify-between mb-2'>
                    <label className='text-sm font-medium text-muted-foreground'>CSA:</label>
                    <Button
                      variant='ghost'
                      size='sm'
                      onClick={() => copyToClipboard(result.csa, 'csa')}
                      className='h-8 w-8 p-0'>
                      {copiedCsa ? (
                        <Check className='h-4 w-4 text-green-500' />
                      ) : (
                        <Copy className='h-4 w-4' />
                      )}
                    </Button>
                  </div>
                  <div className='p-3 bg-muted rounded-md font-mono text-sm whitespace-pre-wrap'>
                    {result.csa}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageSfenizer;
