import React, { useState, useRef, useEffect } from 'react';
import { Rnd } from 'react-rnd';
import { Button } from '@/components/ui/button';
import { Undo2, Redo2, Check, X, Info, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';

interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface BoundingBoxEditorProps {
  imageSrc: string;
  initialTreeBox: number[]; // [x1, y1, x2, y2]
  initialPersonBox: number[]; // [x1, y1, x2, y2]
  currentResults?: { height: string; circumference: string };
  isProcessing?: boolean;
  onLiveUpdate: (treeBox: number[], personBox: number[]) => void;
  onSave: (treeBox: number[], personBox: number[]) => void; // Update this line
  onCancel: () => void;
}

const BoundingBoxEditor: React.FC<BoundingBoxEditorProps> = ({
  imageSrc,
  initialTreeBox,
  initialPersonBox,
  currentResults,
  isProcessing = false,
  onLiveUpdate,
  onSave,
  onCancel,
}) => {
  const [scale, setScale] = useState(1);
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  // Box states store values in NATURAL (original image) coordinates
  const [treeBox, setTreeBox] = useState<Box>(toRndBox(initialTreeBox));
  const [personBox, setPersonBox] = useState<Box>(toRndBox(initialPersonBox));

  // Convert [x1, y1, x2, y2] to {x, y, width, height}
  function toRndBox(box: number[]): Box {
    return {
        x: box[0],
        y: box[1],
        width: box[2] - box[0],
        height: box[3] - box[1],
    };
  }

  // Convert {x, y, width, height} back to [x1, y1, x2, y2]
  function fromRndBox(box: Box): number[] {
      return [
        Math.round(box.x),
        Math.round(box.y),
        Math.round(box.x + box.width),
        Math.round(box.y + box.height),
      ];
  }
  
  // History for Undo/Redo
  const [history, setHistory] = useState<{ tree: Box; person: Box }[]>([
    { tree: toRndBox(initialTreeBox), person: toRndBox(initialPersonBox) }
  ]);
  const [historyIndex, setHistoryIndex] = useState(0);

  // Debounce Logic for Live Update
  useEffect(() => {
    const timer = setTimeout(() => {
        onLiveUpdate(fromRndBox(treeBox), fromRndBox(personBox));
    }, 1000); // 1 second delay

    return () => clearTimeout(timer);
  }, [treeBox, personBox]);

  const handleImageLoad = () => {
    if (imgRef.current) {
      const displayedWidth = imgRef.current.width;
      const naturalWidth = imgRef.current.naturalWidth;
      
      if (naturalWidth > 0) {
        setScale(displayedWidth / naturalWidth);
        setIsImageLoaded(true);
      }
    }
  };

  // Recalculate scale on window resize
  useEffect(() => {
    const handleResize = () => {
        if (imgRef.current) {
             const displayedWidth = imgRef.current.width;
             const naturalWidth = imgRef.current.naturalWidth;
             if (naturalWidth > 0) {
                setScale(displayedWidth / naturalWidth);
             }
        }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const addToHistory = (newTree: Box, newPerson: Box) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({ tree: newTree, person: newPerson });
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const handleUndo = () => {
    if (historyIndex > 0) {
      const prevState = history[historyIndex - 1];
      setTreeBox(prevState.tree);
      setPersonBox(prevState.person);
      setHistoryIndex(historyIndex - 1);
    }
  };

  const handleRedo = () => {
    if (historyIndex < history.length - 1) {
      const nextState = history[historyIndex + 1];
      setTreeBox(nextState.tree);
      setPersonBox(nextState.person);
      setHistoryIndex(historyIndex + 1);
    }
  };

  const resizeHandleStyle = {
      width: '20px',
      height: '20px',
      marginTop: '-10px',
      marginLeft: '-10px',
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <Card className="w-full max-w-4xl h-[90vh] flex flex-col bg-background/95 border-primary/20 shadow-2xl">
        <CardHeader className="flex flex-row items-center justify-between py-3 border-b bg-muted/20">
          <div className="flex items-center gap-4">
              <CardTitle className="text-xl font-bold flex items-center gap-2">
                Chỉnh sửa vùng đo
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Kéo khung màu XANH bao quanh CÂY</p>
                      <p>Kéo khung màu ĐỎ bao quanh NGƯỜI (hoặc vật chuẩn)</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
              
              {/* Live Results Panel */}
              <div className="hidden md:flex items-center gap-3 bg-background/50 px-3 py-1.5 rounded-lg border">
                  {isProcessing ? (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Đang tính toán...
                      </div>
                  ) : (
                      <>
                        <Badge variant="outline" className="text-sm bg-green-500/10 border-green-500/50 text-green-700">
                             Cao: {currentResults?.height || '--'} m
                        </Badge>
                        <Badge variant="outline" className="text-sm bg-blue-500/10 border-blue-500/50 text-blue-700">
                             Chu vi: {currentResults?.circumference || '--'} cm
                        </Badge>
                      </>
                  )}
              </div>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" size="icon" onClick={handleUndo} disabled={historyIndex === 0}>
              <Undo2 className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="icon" onClick={handleRedo} disabled={historyIndex === history.length - 1}>
              <Redo2 className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>
        
        <CardContent className="flex-1 relative overflow-hidden bg-zinc-900 p-0 flex items-center justify-center">
            {/* Mobile Results Overlay (Visible only on small screens) */}
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 flex md:hidden items-center gap-2 bg-background/90 backdrop-blur px-3 py-2 rounded-full shadow-lg border">
                  {isProcessing ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                      <div className="flex gap-3 text-xs font-semibold">
                         <span className="text-green-600">H: {currentResults?.height || '--'}m</span>
                         <div className="w-px h-4 bg-border"></div>
                         <span className="text-blue-600">C: {currentResults?.circumference || '--'}cm</span>
                      </div>
                  )}
            </div>

          <div className="relative inline-block">
             <img 
                ref={imgRef}
                src={imageSrc} 
                onLoad={handleImageLoad}
                alt="Measurement Target" 
                className="max-h-[75vh] max-w-full object-contain select-none pointer-events-none"
                draggable={false}
              />
              
              {isImageLoaded && (
                  <>
                      {/* Tree Box (Green) */}
                      <Rnd
                        size={{ width: treeBox.width * scale, height: treeBox.height * scale }}
                        position={{ x: treeBox.x * scale, y: treeBox.y * scale }}
                        onDragStop={(e, d) => {
                            // Update NATURAL coordinates
                            const newBox = { 
                                ...treeBox, 
                                x: d.x / scale, 
                                y: d.y / scale 
                            };
                            setTreeBox(newBox);
                            addToHistory(newBox, personBox);
                        }}
                        onResizeStop={(e, direction, ref, delta, position) => {
                            // Update NATURAL coordinates
                            const newBox = {
                                width: parseInt(ref.style.width) / scale,
                                height: parseInt(ref.style.height) / scale,
                                x: position.x / scale,
                                y: position.y / scale
                            };
                            setTreeBox(newBox);
                            addToHistory(newBox, personBox);
                        }}
                        bounds="parent"
                        className="border-2 border-green-500 bg-green-500/20 z-10 hover:border-green-400 transition-colors"
                        resizeHandleClasses={{
                            bottomRight: "bg-green-500 rounded-full shadow-sm",
                            bottomLeft: "bg-green-500 rounded-full shadow-sm",
                            topRight: "bg-green-500 rounded-full shadow-sm",
                            topLeft: "bg-green-500 rounded-full shadow-sm",
                        }}
                        resizeHandleStyles={{
                            bottomRight: resizeHandleStyle,
                            bottomLeft: resizeHandleStyle,
                            topRight: resizeHandleStyle,
                            topLeft: resizeHandleStyle,
                        }}
                      >
                        <div className="absolute -top-7 left-0 bg-green-500 text-white text-xs px-2 py-1 rounded shadow-sm font-bold tracking-wide">
                            CÂY
                        </div>
                      </Rnd>

                      {/* Person Box (Red) */}
                      <Rnd
                        size={{ width: personBox.width * scale, height: personBox.height * scale }}
                        position={{ x: personBox.x * scale, y: personBox.y * scale }}
                        onDragStop={(e, d) => {
                            const newBox = { 
                                ...personBox, 
                                x: d.x / scale, 
                                y: d.y / scale 
                            };
                            setPersonBox(newBox);
                            addToHistory(treeBox, newBox);
                        }}
                        onResizeStop={(e, direction, ref, delta, position) => {
                             const newBox = {
                                width: parseInt(ref.style.width) / scale,
                                height: parseInt(ref.style.height) / scale,
                                x: position.x / scale,
                                y: position.y / scale
                            };
                            setPersonBox(newBox);
                            addToHistory(treeBox, newBox);
                        }}
                        bounds="parent"
                        className="border-2 border-red-500 bg-red-500/20 z-20 hover:border-red-400 transition-colors"
                        resizeHandleClasses={{
                            bottomRight: "bg-red-500 rounded-full shadow-sm",
                            bottomLeft: "bg-red-500 rounded-full shadow-sm",
                            topRight: "bg-red-500 rounded-full shadow-sm",
                            topLeft: "bg-red-500 rounded-full shadow-sm",
                        }}
                        resizeHandleStyles={{
                            bottomRight: resizeHandleStyle,
                            bottomLeft: resizeHandleStyle,
                            topRight: resizeHandleStyle,
                            topLeft: resizeHandleStyle,
                        }}
                      >
                        <div className="absolute -top-7 left-0 bg-red-500 text-white text-xs px-2 py-1 rounded shadow-sm font-bold tracking-wide">
                            NGƯỜI
                        </div>
                      </Rnd>
                  </>
              )}
          </div>
        </CardContent>

        <div className="p-4 border-t flex justify-end gap-3 bg-background">
          <Button variant="outline" onClick={onCancel}>
            <X className="w-4 h-4 mr-2" /> Hủy bỏ
          </Button>
          <Button onClick={() => onSave(fromRndBox(treeBox), fromRndBox(personBox))} className="bg-primary hover:bg-primary/90">
            <Check className="w-4 h-4 mr-2" /> Hoàn tất
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default BoundingBoxEditor;
