import { useNavigate } from 'react-router-dom';
import { useMeasurementStore } from '@/store/measurementStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useState, useRef, useEffect } from 'react';
import { Calculator, Ruler, TreePine, MessageCircle, Camera, Loader2, AlertTriangle, Edit } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import BoundingBoxEditor from '@/components/BoundingBoxEditor';
import { useToast } from '@/hooks/use-toast';

const MeasurementScreen = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { selectedTree, setMeasurements, circumference, height } = useMeasurementStore();
  
  const [formData, setFormData] = useState({
    circumference: circumference || '',
    height: height || '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  // State for the new image analysis feature
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // New state for bounding box editing
  const [boundingBoxes, setBoundingBoxes] = useState<{ tree: number[] | null, person: number[] | null }>({ tree: null, person: null });
  const [showEditor, setShowEditor] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isRecalculating, setIsRecalculating] = useState(false);

  // Ref to check if the component is still mounted
  const isMounted = useRef(true);

  useEffect(() => {
    isMounted.current = true;
    // Cleanup function to run when the component unmounts
    return () => {
      isMounted.current = false;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);


  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setImageFile(file);
      setApiError(null);
      setBoundingBoxes({ tree: null, person: null });
      setFormData({ circumference: '', height: '' }); // Reset form
      
      // Create preview URL
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleImageAnalysis = async () => {
    if (!imageFile) {
      setApiError("Vui lòng chọn một file ảnh.");
      return;
    }

    setIsLoading(true);
    setApiError(null);

    const formData = new FormData();
    formData.append("file", imageFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Phân tích ảnh thất bại.");
      }

      const data = await response.json();
      
      // Check if component is still mounted before updating state
      if (isMounted.current) {
        const calculatedCircumference = data.dbh_cm * Math.PI;
        const calculatedHeight = data.tree_height_m;

        if (calculatedHeight > 0) {
          setFormData({
            circumference: calculatedCircumference > 0 ? calculatedCircumference.toFixed(1) : '',
            height: calculatedHeight.toFixed(1),
          });
          
          // Store boxes if available
          if (data.boxes) {
            setBoundingBoxes({
                tree: data.boxes.tree,
                person: data.boxes.person
            });
          }
        } else {
          setApiError("Không thể xác định cây trong ảnh. Vui lòng thử ảnh khác rõ hơn.");
        }
      }

    } catch (error: any) {
      console.error("Lỗi khi gọi API phân tích ảnh:", error);
       if (isMounted.current) {
        setApiError(error.message || "Đã có lỗi xảy ra. Hãy chắc chắn server API đang chạy.");
      }
    } finally {
       if (isMounted.current) {
        setIsLoading(false);
      }
    }
  };

  const handleLiveRecalculate = async (newTreeBox: number[], newPersonBox: number[]) => {
      // Don't close editor, just recalculate
      setIsRecalculating(true);
      // setApiError(null); // Optional: don't clear main error, maybe show local warning

      if (!imageFile) return;

      const formData = new FormData();
      formData.append("file", imageFile);
      formData.append("tree_box", JSON.stringify(newTreeBox));
      formData.append("person_box", JSON.stringify(newPersonBox));

      try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
             throw new Error("Tính toán lại thất bại.");
        }

        const data = await response.json();

        if (isMounted.current) {
             const calculatedCircumference = data.dbh_cm * Math.PI;
             const calculatedHeight = data.tree_height_m;

             setFormData({
                circumference: calculatedCircumference > 0 ? calculatedCircumference.toFixed(1) : '',
                height: calculatedHeight.toFixed(1),
             });
             
             // Silent update of boxes not strictly needed as we are driving from UI, 
             // but good for sync if we were to close/reopen
             if (data.boxes) {
                setBoundingBoxes({
                    tree: data.boxes.tree,
                    person: data.boxes.person
                });
             }
        }
      } catch (error: any) {
         console.error("Lỗi background recalc:", error);
         // Handle silently or show toast
      } finally {
         if (isMounted.current) {
             setIsRecalculating(false);
         }
      }
  };

  const handleChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    setErrors((prev) => ({ ...prev, [field]: '' }));
  };

  const validate = () => {
    const newErrors: Record<string, string> = {};
    
    const circ = parseFloat(formData.circumference as string);
    const h = parseFloat(formData.height as string);
    
    if (!formData.circumference || isNaN(circ) || circ <= 0) {
      newErrors.circumference = 'Vui lòng nhập chu vi hợp lệ';
    }
    if (!formData.height || isNaN(h) || h <= 0 || h > 100) {
      newErrors.height = 'Chiều cao phải từ 0.1 đến 100 mét';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validate()) return;
    
    setMeasurements(
      parseFloat(formData.circumference as string),
      parseFloat(formData.height as string)
    );
    
    navigate('/loading');
  };
  
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      {showEditor && previewUrl && boundingBoxes.tree && boundingBoxes.person && (
          <BoundingBoxEditor
            imageSrc={previewUrl}
            initialTreeBox={boundingBoxes.tree}
            initialPersonBox={boundingBoxes.person}
            currentResults={{
                height: formData.height as string,
                circumference: formData.circumference as string
            }}
            isProcessing={isRecalculating}
            onLiveUpdate={handleLiveRecalculate}
            onSave={(finalTree, finalPerson) => {
                // Trigger final recalculate in background
                handleLiveRecalculate(finalTree, finalPerson).then(() => {
                    toast({
                        title: "Đã cập nhật dữ liệu",
                        description: "Kết quả đo từ ảnh đã được điền vào form.",
                    });
                });
                // Close editor immediately for snappy feel
                setShowEditor(false);
            }}
            onCancel={() => setShowEditor(false)}
          />
      )}

      <div className="glass-card p-6 md:p-8 w-full max-w-md animate-fade-in">
        <div className="text-center mb-6">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary/20 flex items-center justify-center border border-primary/30 animate-scale-in">
            <Ruler className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">Nhập Dữ Liệu Đo</h1>
          {selectedTree && (
            <p className="text-muted-foreground mt-2">
              Cây số <span className="text-primary font-semibold">{selectedTree.treeNumber}</span>
              {' - '}{selectedTree.species}
            </p>
          )}
        </div>

        {/* AI Measurement Section */}
        <Card className="mb-6 bg-muted/30 border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="w-5 h-5 text-primary" />
              Đo giúp tôi
            </CardTitle>
            <CardDescription>Tải ảnh lên để AI tự động ước tính</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="image-upload">Chọn ảnh</Label>
              <Input 
                id="image-upload" 
                type="file" 
                accept="image/*"
                onChange={handleFileChange}
                className="file:text-primary-foreground"
              />
            </div>
            {apiError && (
              <div className="flex items-center gap-2 text-destructive text-sm p-2 bg-destructive/10 rounded-md">
                <AlertTriangle className="w-4 h-4" />
                <p>{apiError}</p>
              </div>
            )}
            <div className="flex gap-2">
                <Button 
                  onClick={handleImageAnalysis} 
                  disabled={isLoading || !imageFile}
                  className="flex-1"
                >
                  {isLoading ? (
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  ) : (
                    <TreePine className="w-5 h-5 mr-2" />
                  )}
                  {isLoading ? 'Đang phân tích...' : 'Phân tích ảnh'}
                </Button>
                
                {boundingBoxes.tree && boundingBoxes.person && !isLoading && (
                    <Button
                        variant="outline"
                        onClick={() => setShowEditor(true)}
                        className="w-12 px-0"
                        title="Chỉnh sửa vùng đo"
                    >
                        <Edit className="w-5 h-5 text-primary" />
                    </Button>
                )}
            </div>
          </CardContent>
        </Card>

        {/* Manual Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="circumference" className="text-foreground flex items-center gap-2">
              <div className="w-6 h-6 rounded-full border-2 border-current flex items-center justify-center text-xs">C</div>
              Chu vi thân cây (cm)
            </Label>
            <Input
              id="circumference"
              type="number"
              step="0.1"
              placeholder="Ví dụ: 120"
              value={formData.circumference}
              onChange={(e) => handleChange('circumference', e.target.value)}
              className="bg-input/50 border-border/50 focus:border-primary text-foreground placeholder:text-muted-foreground"
            />
            {errors.circumference && <p className="text-destructive text-sm">{errors.circumference}</p>}
          </div>

          <div className="space-y-2">
            <Label htmlFor="height" className="text-foreground flex items-center gap-2">
              <TreePine className="w-5 h-5" />
              Chiều cao cây (m)
            </Label>
            <Input
              id="height"
              type="number"
              step="0.1"
              placeholder="Ví dụ: 12.5"
              value={formData.height}
              onChange={(e) => handleChange('height', e.target.value)}
              className="bg-input/50 border-border/50 focus:border-primary text-foreground placeholder:text-muted-foreground"
            />
            {errors.height && <p className="text-destructive text-sm">{errors.height}</p>}
          </div>

          <Button
            type="submit"
            className="w-full h-12 text-lg font-semibold bg-primary hover:bg-primary/90 text-primary-foreground glow-primary transition-all duration-300"
          >
            <Calculator className="w-5 h-5 mr-2" />
            Tính toán kết quả
          </Button>
        </form>

        <div className="mt-4">
          <Button
            type="button"
            variant="outline"
            onClick={() => navigate('/chat')}
            className="w-full gap-2 border-primary/30 hover:bg-primary/10 hover:border-primary/50"
          >
            <MessageCircle className="w-4 h-4" />
            Cần hướng dẫn đo cây?
          </Button>
        </div>
      </div>
    </div>
  );
};

export default MeasurementScreen;
