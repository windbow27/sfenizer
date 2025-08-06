import React, { useState, useRef } from 'react';
import { Camera, Upload, Image, X, Check, Copy } from 'lucide-react';
import { Button } from './components/ui/button';
import { Card, CardContent } from './components/ui/card';
import { toast } from 'sonner';

interface FileHandler {
  (file: File): void;
}

interface ConversionResult {
  success: boolean;
  sfen: string;
  csa: string;
  board: string[][];
}

const App: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ConversionResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const [copiedSfen, setCopiedSfen] = useState<boolean>(false);
  const [copiedCsa, setCopiedCsa] = useState<boolean>(false);

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
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        if (e.target?.result) {
          setSelectedImage(e.target.result as string);
        }
      };
      reader.readAsDataURL(file);
      setResult(null);
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

  const handleConvert = async (): Promise<void> => {
    if (!selectedFile) {
      toast.error('No image selected');
      return;
    }

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const API_BASE_URL = import.meta.env.PROD ? '/api' : 'http://localhost:8000';

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
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const clearImage = (): void => {
    setSelectedImage(null);
    setSelectedFile(null);
    setResult(null);
  };

  return (
    <div className='min-h-screen bg-background'>
      {/* Navbar */}
      <nav className='border-b border-border bg-card'>
        <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
          <div className='flex h-16 items-center justify-between'>
            <div className='flex items-center'>
              <h1 className='text-xl font-semibold text-foreground'>Sfenizer</h1>
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
                      <p className='text-lg font-medium text-foreground'>
                        Drop your shogi board image here
                      </p>
                      <p className='text-sm text-muted-foreground'>Or just Ctrl +V, all works</p>
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
            <Button
              onClick={handleConvert}
              disabled={!selectedFile || isLoading}
              size='lg'
              className='px-8'>
              {isLoading ? 'Converting...' : 'Convert to SFEN/CSA'}
            </Button>
          </div>

          {/* Results Display */}
          {result && (
            <Card>
              <CardContent className='p-6'>
                <div className='space-y-4'>
                  <h3 className='text-lg font-semibold'>Conversion Results</h3>

                  <div className='space-y-3'>
                    <div>
                      <div className='flex items-center justify-between mb-2'>
                        <label className='text-sm font-medium text-muted-foreground'>SFEN:</label>
                        <Button
                          variant='ghost'
                          size='sm'
                          onClick={() => copyToClipboard(result.sfen, 'sfen')}
                          className='h-8 w-8 p-0'>
                          {copiedSfen ? (
                            <Check className='h-4 w-4 text-success' />
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
                            <Check className='h-4 w-4 text-success' />
                          ) : (
                            <Copy className='h-4 w-4' />
                          )}
                        </Button>
                      </div>
                      <div className='p-3 bg-muted rounded-md font-mono text-sm whitespace-pre-wrap'>
                        {result.csa}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
