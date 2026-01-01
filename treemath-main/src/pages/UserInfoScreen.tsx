import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useMeasurementStore } from '@/store/measurementStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useState } from 'react';
import { ArrowRight, User, School } from 'lucide-react';
import FallingLeaves from '@/components/FallingLeaves';

const UserInfoScreen = () => {
  const navigate = useNavigate();
  const { setUserInfo, selectedTree, userName, userClass } = useMeasurementStore();
  const [name, setName] = useState(userName);
  const [className, setClassName] = useState(userClass);
  const [errors, setErrors] = useState<{ name?: string; className?: string }>({});

  // Validation constants
  const MAX_NAME_LENGTH = 50;
  const MAX_CLASS_LENGTH = 10;
  const NAME_PATTERN = /^[\p{L}\p{N}\s\-''.]+$/u;
  const CLASS_PATTERN = /^[0-9]{1,2}[A-Za-z][0-9]{0,2}$/;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const trimmedName = name.trim();
    const trimmedClass = className.trim();
    
    const newErrors: { name?: string; className?: string } = {};
    
    // Name validation
    if (!trimmedName) {
      newErrors.name = 'Vui lòng nhập họ tên';
    } else if (trimmedName.length > MAX_NAME_LENGTH) {
      newErrors.name = `Họ tên không được quá ${MAX_NAME_LENGTH} ký tự`;
    } else if (!NAME_PATTERN.test(trimmedName)) {
      newErrors.name = 'Họ tên chỉ được chứa chữ cái, số và khoảng trắng';
    }
    
    // Class validation
    if (!trimmedClass) {
      newErrors.className = 'Vui lòng nhập lớp';
    } else if (trimmedClass.length > MAX_CLASS_LENGTH) {
      newErrors.className = `Lớp không được quá ${MAX_CLASS_LENGTH} ký tự`;
    } else if (!CLASS_PATTERN.test(trimmedClass)) {
      newErrors.className = 'Định dạng lớp không hợp lệ (ví dụ: 10A1)';
    }
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    setUserInfo(trimmedName, trimmedClass);
    navigate('/measurement');
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden flex items-center justify-center p-4">
      {/* Falling leaves background */}
      <FallingLeaves />

      {/* Background tree silhouette */}
      <div className="absolute inset-0 flex items-center justify-center opacity-20 pointer-events-none">
        <svg viewBox="0 0 200 300" className="w-full h-full max-w-md">
          <rect x="85" y="200" width="30" height="100" className="fill-primary/30" />
          <ellipse cx="100" cy="180" rx="80" ry="50" className="fill-primary/20" />
          <ellipse cx="100" cy="130" rx="65" ry="45" className="fill-primary/25" />
          <ellipse cx="100" cy="85" rx="50" ry="35" className="fill-primary/20" />
          <ellipse cx="100" cy="50" rx="30" ry="25" className="fill-primary/25" />
        </svg>
      </div>

      {/* Glass card form */}
      <motion.div
        className="glass-card p-6 md:p-8 w-full max-w-md relative z-10"
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className="text-center mb-6">
          <motion.div
            className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary/20 
                       flex items-center justify-center border border-primary/30"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring" }}
          >
            <User className="w-8 h-8 text-primary" />
          </motion.div>
          
          <h1 className="text-2xl font-bold text-foreground">Thông Tin Của Bạn</h1>
          
          {selectedTree && (
            <p className="text-muted-foreground mt-2">
              Đo cây số <span className="text-primary font-semibold">{selectedTree.treeNumber}</span>
              {' - '}{selectedTree.species}
            </p>
          )}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="name" className="text-foreground flex items-center gap-2">
              <User className="w-4 h-4" />
              Họ và tên
            </Label>
            <Input
              id="name"
              type="text"
              placeholder="Nhập họ tên của bạn"
              value={name}
              maxLength={50}
              onChange={(e) => {
                setName(e.target.value);
                setErrors((prev) => ({ ...prev, name: undefined }));
              }}
              className="bg-input/50 border-border/50 focus:border-primary 
                         text-foreground placeholder:text-muted-foreground"
            />
            {errors.name && (
              <p className="text-destructive text-sm">{errors.name}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="class" className="text-foreground flex items-center gap-2">
              <School className="w-4 h-4" />
              Lớp
            </Label>
            <Input
              id="class"
              type="text"
              placeholder="Ví dụ: 10A1"
              value={className}
              maxLength={10}
              onChange={(e) => {
                setClassName(e.target.value);
                setErrors((prev) => ({ ...prev, className: undefined }));
              }}
              className="bg-input/50 border-border/50 focus:border-primary 
                         text-foreground placeholder:text-muted-foreground"
            />
            {errors.className && (
              <p className="text-destructive text-sm">{errors.className}</p>
            )}
          </div>

          <Button
            type="submit"
            className="w-full h-12 text-lg font-semibold bg-primary hover:bg-primary/90 
                       text-primary-foreground glow-primary transition-all duration-300"
          >
            Tiếp tục
            <ArrowRight className="w-5 h-5 ml-2" />
          </Button>
        </form>
      </motion.div>
    </div>
  );
};

export default UserInfoScreen;
