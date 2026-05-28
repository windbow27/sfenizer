import React, { useState, useRef } from 'react';
import { Camera, Upload, Image, X, Check, Copy } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Label } from '../components/ui/label';
import { Separator } from '../components/ui/separator';
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
      toast.success(`${type.toUpperCase()} copied!`);
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

      const response = await fetch(`${API_BASE_URL}/convert`, { method: 'POST', body: formData });

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
    <div className='py-8 max-w-3xl'>
      <div className='space-y-6'>
        <div>
          <h1 className='text-xl font-bold'>Image Sfenizer</h1>
          <p className='text-sm text-muted-foreground'>
            Upload a shogi board photo to extract SFEN/CSA notation
          </p>
        </div>

        <div
          className={`rounded-md border-2 border-dashed p-8 text-center outline-none ${dragActive || selectedImage ? 'border-primary' : 'border-border'}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onPaste={handlePaste}
          tabIndex={0}>
          {selectedImage ? (
            <div className='relative inline-block'>
              <img
                src={selectedImage}
                alt='Selected'
                className='mx-auto max-h-64 rounded-md object-contain'
              />
              <Button
                variant='secondary'
                size='sm'
                className='absolute -top-2 -right-2 h-6 w-6 rounded-full p-0'
                onClick={clearImage}>
                <X className='h-3 w-3' />
              </Button>
            </div>
          ) : (
            <div className='space-y-3'>
              <Image className='h-8 w-8 mx-auto text-muted-foreground' />
              <p className='text-sm'>Drop your shogi board image here</p>
              <div className='flex flex-wrap justify-center gap-2'>
                <Button variant='outline' size='sm' onClick={() => fileInputRef.current?.click()}>
                  <Upload className='h-4 w-4 mr-2' />
                  Select File
                </Button>
                <Button variant='outline' size='sm' onClick={() => cameraInputRef.current?.click()}>
                  <Camera className='h-4 w-4 mr-2' />
                  Camera
                </Button>
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

        <div className='flex justify-center'>
          <Button onClick={handleConvert} disabled={!selectedFile || isLoading} size='lg'>
            {isLoading ? 'Converting…' : 'Convert to SFEN/CSA'}
          </Button>
        </div>

        {result && (
          <div className='space-y-6'>
            <div>
              <p className='text-sm font-medium mb-3 flex items-center gap-2'>
                <Check className='h-4 w-4 text-primary' />
                Detected Board
              </p>
              <ShogiBoard sfen={result.sfen} maxWidth={520} />
            </div>

            <Separator />

            <div className='space-y-4'>
              <div className='space-y-1'>
                <div className='flex items-center justify-between'>
                  <Label>SFEN</Label>
                  <Button
                    variant='ghost'
                    size='sm'
                    className='h-7 w-7 p-0'
                    onClick={() => copyToClipboard(result.sfen, 'sfen')}>
                    {copiedSfen ? (
                      <Check className='h-3 w-3 text-green-500' />
                    ) : (
                      <Copy className='h-3 w-3' />
                    )}
                  </Button>
                </div>
                <pre className='p-3 bg-muted rounded-md text-xs break-all whitespace-pre-wrap font-mono'>
                  {result.sfen}
                </pre>
              </div>

              <div className='space-y-1'>
                <div className='flex items-center justify-between'>
                  <Label>CSA</Label>
                  <Button
                    variant='ghost'
                    size='sm'
                    className='h-7 w-7 p-0'
                    onClick={() => copyToClipboard(result.csa, 'csa')}>
                    {copiedCsa ? (
                      <Check className='h-3 w-3 text-green-500' />
                    ) : (
                      <Copy className='h-3 w-3' />
                    )}
                  </Button>
                </div>
                <pre className='p-3 bg-muted rounded-md text-xs whitespace-pre-wrap font-mono'>
                  {result.csa}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageSfenizer;
