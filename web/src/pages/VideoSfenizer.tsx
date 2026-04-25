import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Copy, Check, RefreshCw } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { toast } from 'sonner';

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/ws/video`;

interface LiveResult {
  frame: string;
  sfen: string;
  csa: string;
}

const VideoSfenizer: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [liveResult, setLiveResult] = useState<LiveResult | null>(null);
  const [copiedSfen, setCopiedSfen] = useState(false);
  const [copiedCsa, setCopiedCsa] = useState(false);
  // toggle between video preview and annotated frame
  const [showAnnotated, setShowAnnotated] = useState(true);

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

  const sendFrame = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const MAX_DIM = 720;
    const scale = Math.min(1, MAX_DIM / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64 = canvas.toDataURL('image/jpeg', 0.75).split(',')[1];
    wsRef.current.send(base64);
  }, []);

  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsRunning(false);
    setIsConnecting(false);
  }, []);

  const startCamera = async () => {
    setIsConnecting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsRunning(true);
        setIsConnecting(false);
        // Send a frame every 300ms (≈3fps) — enough for real-time without overloading
        intervalRef.current = setInterval(sendFrame, 300);
      };

      ws.onmessage = (event) => {
        const data: LiveResult = JSON.parse(event.data);
        setLiveResult(data);
      };

      ws.onerror = () => {
        toast.error('WebSocket connection failed — is the backend running?');
        stopCamera();
      };

      ws.onclose = () => {
        stopCamera();
      };
    } catch (err) {
      toast.error(
        'Camera access denied: ' + (err instanceof Error ? err.message : 'Unknown error')
      );
      stopCamera();
    }
  };

  // Clean up on unmount
  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  return (
    <div className='py-4 md:py-10 max-w-2xl page-enter'>
      <div className='space-y-6'>
        <div className='flex items-center gap-3 animate-fade-up'>
          <div className='h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center'>
            <Camera className='h-5 w-5 text-primary' />
          </div>
          <div>
            <h1 className='text-xl font-bold text-foreground'>Video Sfenizer</h1>
            <p className='text-xs text-muted-foreground'>
              Real-time shogi board detection — point your camera at the board
            </p>
          </div>
        </div>

        {/* Camera / annotated feed */}
        <Card className='-mx-4 sm:mx-0 overflow-hidden border-border/60 animate-fade-up delay-100'>
          <CardContent className='p-0'>
            <div className='relative bg-black sm:rounded-lg overflow-hidden min-h-[40vw]'>
              {/* Live camera preview (always rendered so the stream stays alive) */}
              <video
                ref={videoRef}
                playsInline
                muted
                className={`w-full block ${showAnnotated && liveResult ? 'hidden' : ''}`}
              />

              {/* Annotated frame from backend */}
              {showAnnotated && liveResult && (
                <img
                  src={`data:image/jpeg;base64,${liveResult.frame}`}
                  alt='Annotated'
                  className='w-full block'
                />
              )}

              {/* Placeholder when camera is off */}
              {!isRunning && !isConnecting && (
                <div className='absolute inset-0 flex flex-col items-center justify-center text-muted-foreground bg-muted/30'>
                  <CameraOff className='h-12 w-12 mb-3 opacity-30' />
                  <p className='text-sm'>Camera not started</p>
                </div>
              )}

              {isConnecting && (
                <div className='absolute inset-0 flex flex-col items-center justify-center text-muted-foreground'>
                  <RefreshCw className='h-10 w-10 mb-3 animate-spin opacity-60' />
                  <p className='text-sm'>Connecting…</p>
                </div>
              )}

              {/* Toggle overlay */}
              {isRunning && liveResult && (
                <button
                  onClick={() => setShowAnnotated((prev) => !prev)}
                  className='absolute top-2 right-2 rounded-md bg-black/50 px-2 py-1 text-xs text-white hover:bg-black/70 transition-colors'>
                  {showAnnotated ? 'Show Raw' : 'Show Detected'}
                </button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} className='hidden' />

        {/* Controls */}
        <div className='flex justify-center animate-fade-up delay-200'>
          <Button
            onClick={isRunning ? stopCamera : startCamera}
            disabled={isConnecting}
            size='lg'
            variant={isRunning ? 'destructive' : 'default'}
            className='gap-2 px-8 transition-transform duration-200 hover:scale-105 active:scale-95'>
            {isRunning ? (
              <>
                <CameraOff className='h-5 w-5' />
                Stop Camera
              </>
            ) : (
              <>
                <Camera className='h-5 w-5' />
                {isConnecting ? 'Connecting…' : 'Start Camera'}
              </>
            )}
          </Button>
        </div>

        {/* Live SFEN / CSA */}
        {liveResult && (
          <Card className='border-primary/20 animate-scale-in'>
            <CardContent className='p-5 sm:p-6 space-y-4'>
              <h3 className='text-base font-semibold flex items-center gap-2'>
                <span className='h-2 w-2 rounded-full bg-green-500 animate-pulse' />
                Live Detection
              </h3>

              <div>
                <div className='flex items-center justify-between mb-2'>
                  <label className='text-sm font-medium text-muted-foreground'>SFEN:</label>
                  <Button
                    variant='ghost'
                    size='sm'
                    onClick={() => copyToClipboard(liveResult.sfen, 'sfen')}
                    className='h-8 w-8 p-0'>
                    {copiedSfen ? (
                      <Check className='h-4 w-4 text-green-500' />
                    ) : (
                      <Copy className='h-4 w-4' />
                    )}
                  </Button>
                </div>
                <div className='p-3 bg-muted rounded-md font-mono text-sm break-all'>
                  {liveResult.sfen}
                </div>
              </div>

              <div>
                <div className='flex items-center justify-between mb-2'>
                  <label className='text-sm font-medium text-muted-foreground'>CSA:</label>
                  <Button
                    variant='ghost'
                    size='sm'
                    onClick={() => copyToClipboard(liveResult.csa, 'csa')}
                    className='h-8 w-8 p-0'>
                    {copiedCsa ? (
                      <Check className='h-4 w-4 text-green-500' />
                    ) : (
                      <Copy className='h-4 w-4' />
                    )}
                  </Button>
                </div>
                <div className='p-3 bg-muted rounded-md font-mono text-sm whitespace-pre-wrap'>
                  {liveResult.csa}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default VideoSfenizer;
