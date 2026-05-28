import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Copy, Check, RefreshCw } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Separator } from '../components/ui/separator';
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
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsRunning(false);
    setIsConnecting(false);
  }, []);

  const startCamera = async () => {
    setIsConnecting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
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
        intervalRef.current = setInterval(sendFrame, 300);
      };
      ws.onmessage = (event) => setLiveResult(JSON.parse(event.data));
      ws.onerror = () => {
        toast.error('WebSocket connection failed — is the backend running?');
        stopCamera();
      };
      ws.onclose = () => stopCamera();
    } catch (err) {
      toast.error(
        'Camera access denied: ' + (err instanceof Error ? err.message : 'Unknown error')
      );
      stopCamera();
    }
  };

  useEffect(() => () => stopCamera(), [stopCamera]);

  return (
    <div className='py-8 max-w-2xl'>
      <div className='space-y-6'>
        <div>
          <h1 className='text-xl font-bold'>Video Sfenizer</h1>
          <p className='text-sm text-muted-foreground'>Real-time shogi board detection</p>
        </div>

        <div className='relative bg-black sm:rounded-lg overflow-hidden min-h-[40vw]'>
          <video
            ref={videoRef}
            playsInline
            muted
            className={`w-full block ${showAnnotated && liveResult ? 'hidden' : ''}`}
          />

          {showAnnotated && liveResult && (
            <img
              src={`data:image/jpeg;base64,${liveResult.frame}`}
              alt='Annotated'
              className='w-full block'
            />
          )}

          {!isRunning && !isConnecting && (
            <div className='absolute inset-0 flex flex-col items-center justify-center text-muted-foreground'>
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

          {isRunning && liveResult && (
            <button
              onClick={() => setShowAnnotated((prev) => !prev)}
              className='absolute top-2 right-2 rounded-md bg-black/50 px-2 py-1 text-xs text-white'>
              {showAnnotated ? 'Show Raw' : 'Show Detected'}
            </button>
          )}
        </div>

        <canvas ref={canvasRef} className='hidden' />

        <div className='flex justify-center'>
          <Button
            onClick={isRunning ? stopCamera : startCamera}
            disabled={isConnecting}
            size='lg'
            variant={isRunning ? 'destructive' : 'default'}>
            {isRunning ? (
              <>
                <CameraOff className='h-5 w-5 mr-2' />
                Stop Camera
              </>
            ) : (
              <>
                <Camera className='h-5 w-5 mr-2' />
                {isConnecting ? 'Connecting…' : 'Start Camera'}
              </>
            )}
          </Button>
        </div>

        {liveResult && (
          <div className='space-y-4'>
            <div className='flex items-center gap-2'>
              <span className='h-2 w-2 rounded-full bg-green-500 animate-pulse' />
              <p className='text-sm font-medium'>Live Detection</p>
            </div>

            <Separator />

            <div className='space-y-1'>
              <div className='flex items-center justify-between'>
                <Label>SFEN</Label>
                <Button
                  variant='ghost'
                  size='sm'
                  className='h-7 w-7 p-0'
                  onClick={() => copyToClipboard(liveResult.sfen, 'sfen')}>
                  {copiedSfen ? (
                    <Check className='h-3 w-3 text-green-500' />
                  ) : (
                    <Copy className='h-3 w-3' />
                  )}
                </Button>
              </div>
              <pre className='p-3 bg-muted rounded-md text-xs break-all whitespace-pre-wrap font-mono'>
                {liveResult.sfen}
              </pre>
            </div>

            <div className='space-y-1'>
              <div className='flex items-center justify-between'>
                <Label>CSA</Label>
                <Button
                  variant='ghost'
                  size='sm'
                  className='h-7 w-7 p-0'
                  onClick={() => copyToClipboard(liveResult.csa, 'csa')}>
                  {copiedCsa ? (
                    <Check className='h-3 w-3 text-green-500' />
                  ) : (
                    <Copy className='h-3 w-3' />
                  )}
                </Button>
              </div>
              <pre className='p-3 bg-muted rounded-md text-xs whitespace-pre-wrap font-mono'>
                {liveResult.csa}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoSfenizer;
