import { useNavigate } from 'react-router-dom';
import { useMeasurementStore } from '@/store/measurementStore';
import { getCO2Equivalents } from '@/lib/calculations';
import { supabase } from '@/integrations/supabase/client';
import { Button } from '@/components/ui/button';
import { useEffect, useState, useRef } from 'react';
import { TreePine, Cloud, Car, Smartphone, Recycle, Trophy, Ruler, Circle } from 'lucide-react';
import { toast } from 'sonner';
import { motion } from 'framer-motion';

// Simple rate limiting: max 5 submissions per minute per session (client-side backup)
const RATE_LIMIT_WINDOW = 60000; // 1 minute
const MAX_SUBMISSIONS = 5;

// Animation variants - simpler without transition in variants
const fadeInUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0 }
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.85, y: 20 },
  visible: { opacity: 1, scale: 1, y: 0 }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.4,
      delayChildren: 0.3
    }
  }
};

const slideInFromLeft = {
  hidden: { opacity: 0, x: -50 },
  visible: { opacity: 1, x: 0 }
};

// Animated counter component
const AnimatedNumber = ({ value, suffix = '' }: { value: number; suffix?: string }) => {
  const [displayValue, setDisplayValue] = useState(0);
  
  useEffect(() => {
    const duration = 1500;
    const startTime = Date.now();
    const startValue = 0;
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function (ease-out)
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = startValue + (value - startValue) * easeOut;
      
      setDisplayValue(current);
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    requestAnimationFrame(animate);
  }, [value]);
  
  const formattedValue = Number.isInteger(value) 
    ? Math.round(displayValue).toLocaleString()
    : displayValue.toFixed(1);
  
  return <>{formattedValue}{suffix}</>;
};

const ResultsScreen = () => {
  const navigate = useNavigate();
  const { result, selectedTree, userName, userClass, circumference, height, reset } = useMeasurementStore();
  const [saved, setSaved] = useState(false);
  const [countdown, setCountdown] = useState(10);
  const submissionTimestamps = useRef<number[]>([]);

  // Auto-navigate to leaderboard after countdown
  useEffect(() => {
    if (!selectedTree?.treeNumber) return;

    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          navigate(`/leaderboard/${selectedTree.treeNumber}`);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [navigate, selectedTree?.treeNumber]);

  useEffect(() => {
    const saveResult = async () => {
      if (!result || !selectedTree || saved) return;

      // Client-side rate limiting backup (server enforces the real limit)
      const now = Date.now();
      submissionTimestamps.current = submissionTimestamps.current.filter(
        (timestamp) => now - timestamp < RATE_LIMIT_WINDOW
      );
      
      if (submissionTimestamps.current.length >= MAX_SUBMISSIONS) {
        toast.error('Quá nhiều lần gửi. Vui lòng thử lại sau 1 phút.');
        return;
      }
      
      submissionTimestamps.current.push(now);

      try {
        // Submit measurement via secure edge function
        const { data, error } = await supabase.functions.invoke('submit-measurement', {
          body: {
            tree_id: selectedTree.treeNumber,
            user_name: userName.slice(0, 50),
            user_class: userClass.slice(0, 10),
            measured_circumference: circumference,
            measured_height: height,
            calculated_height: result.calculatedHeight,
            calculated_diameter: result.calculatedDiameter,
            accuracy_score: result.overallAccuracy,
            biomass_kg: result.biomassKg,
            co2_absorbed_kg: result.co2AbsorbedKg,
          },
        });

        if (error) throw error;
        
        if (data?.error) {
          throw new Error(data.error);
        }

        setSaved(true);
      } catch (error) {
        // Only log detailed errors in development
        if (import.meta.env.DEV) {
          console.error('Error saving result:', error);
        }
        
        // Show user-friendly error message
        const errorMessage = error instanceof Error && error.message.includes('Rate limit')
          ? 'Quá nhiều lần gửi. Vui lòng thử lại sau.'
          : 'Không thể lưu kết quả. Vui lòng thử lại.';
        toast.error(errorMessage);
      }
    };

    saveResult();
  }, [result, selectedTree, userName, userClass, circumference, height, saved]);

  // Mock data for development preview
  const displayResult = result || {
    calculatedHeight: 12.5,
    calculatedDiameter: 35.2,
    heightAccuracy: 87.5,
    diameterAccuracy: 92.3,
    overallAccuracy: 89.9,
    biomassKg: 245.8,
    co2AbsorbedKg: 451.2,
  };

  const displayTree = selectedTree || {
    treeNumber: 1,
    species: 'Sao đen',
    actualHeight: 14,
    actualDiameter: 38,
  };

  const equivalents = getCO2Equivalents(displayResult.co2AbsorbedKg);

  const handlePlayAgain = () => {
    reset();
    navigate('/');
  };

  const handleViewLeaderboard = () => {
    navigate(`/leaderboard/${displayTree.treeNumber}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-950 via-green-900/80 to-background p-4 overflow-auto">
      <div className="max-w-5xl mx-auto">
        {/* Countdown timer */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center text-sm text-muted-foreground mb-4"
        >
          Chuyển sang bảng xếp hạng sau {countdown}s
        </motion.div>

        {/* Header */}
        <motion.div 
          variants={fadeInUp}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.8, delay: 0.3 }}
          className="text-center mb-6"
        >
          <h1 className="text-2xl md:text-4xl font-bold text-foreground drop-shadow-lg">
            🌳 Kết Quả Đo Đạc
          </h1>
          <p className="text-muted-foreground mt-2 text-lg">
            Cây số {displayTree.treeNumber} - {displayTree.species}
          </p>
        </motion.div>

        {/* Main content - Responsive grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column */}
          <div className="space-y-6">
            {/* Measurement results */}
            <motion.div 
              variants={scaleIn}
              initial="hidden"
              animate="visible"
              transition={{ duration: 0.8, delay: 0.8 }}
              className="glass-card p-6 glow-primary"
            >
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <Ruler className="w-5 h-5 text-primary" />
                Kết Quả Đo
              </h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-primary/20 rounded-xl border border-primary/30">
                  <Ruler className="w-8 h-8 mx-auto mb-2 text-primary" />
                  <p className="text-sm text-muted-foreground">Chiều cao</p>
                  <p className="text-3xl font-bold text-primary">
                    <AnimatedNumber value={displayResult.calculatedHeight} suffix="m" />
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Thực tế: {displayTree.actualHeight}m
                  </p>
                </div>
                <div className="text-center p-4 bg-accent/20 rounded-xl border border-accent/30">
                  <Circle className="w-8 h-8 mx-auto mb-2 text-accent" />
                  <p className="text-sm text-muted-foreground">Đường kính</p>
                  <p className="text-3xl font-bold text-accent">
                    <AnimatedNumber value={displayResult.calculatedDiameter} suffix="cm" />
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Thực tế: {displayTree.actualDiameter}cm
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Biomass & CO2 */}
            <motion.div 
              variants={scaleIn}
              initial="hidden"
              animate="visible"
              transition={{ duration: 0.8, delay: 1.4 }}
              className="glass-card p-6"
            >
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <TreePine className="w-5 h-5 text-primary" />
                Sinh Khối & CO₂
              </h2>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-gradient-to-br from-primary/20 to-primary/5 rounded-xl border border-primary/30">
                  <TreePine className="w-10 h-10 mx-auto mb-2 text-primary" />
                  <p className="text-sm text-muted-foreground">Sinh khối</p>
                  <p className="text-2xl md:text-3xl font-bold text-foreground">
                    <AnimatedNumber value={displayResult.biomassKg} suffix=" kg" />
                  </p>
                </div>
                <div className="text-center p-4 bg-gradient-to-br from-accent/20 to-accent/5 rounded-xl border border-accent/30">
                  <Cloud className="w-10 h-10 mx-auto mb-2 text-accent" />
                  <p className="text-sm text-muted-foreground">CO₂ hấp thụ</p>
                  <p className="text-2xl md:text-3xl font-bold text-foreground">
                    <AnimatedNumber value={displayResult.co2AbsorbedKg} suffix=" kg" />
                  </p>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Right column - Equivalents */}
          <motion.div 
            variants={scaleIn}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.8, delay: 2.0 }}
            className="rounded-2xl bg-gradient-to-br from-primary/15 via-accent/10 to-primary/5 p-6 border border-primary/30 h-fit lg:h-full glow-primary"
          >
            <h2 className="text-xl font-bold text-primary mb-5 flex items-center gap-2">
              <Recycle className="w-6 h-6" />
              Quy Đổi Tương Đương
            </h2>
            
            <motion.div 
              variants={staggerContainer}
              initial="hidden"
              animate="visible"
              className="space-y-4"
            >
              <motion.div 
                variants={fadeInUp}
                className="flex items-center gap-4 p-4 bg-background/80 rounded-xl border border-border transition-all hover:translate-x-2 hover:shadow-lg hover:shadow-orange-500/20"
              >
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center shadow-lg shadow-orange-500/30">
                  <Car className="w-8 h-8 text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-3xl font-bold text-foreground">
                    <AnimatedNumber value={equivalents.carDays} /> <span className="text-lg font-medium">ngày</span>
                  </p>
                  <p className="text-sm text-muted-foreground">khí thải xe hơi được bù đắp</p>
                </div>
              </motion.div>
              
              <motion.div 
                variants={fadeInUp}
                className="flex items-center gap-4 p-4 bg-background/80 rounded-xl border border-border transition-all hover:translate-x-2 hover:shadow-lg hover:shadow-blue-500/20"
              >
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-blue-500/30">
                  <Smartphone className="w-8 h-8 text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-3xl font-bold text-foreground">
                    <AnimatedNumber value={equivalents.phoneCharges} /> <span className="text-lg font-medium">lần</span>
                  </p>
                  <p className="text-sm text-muted-foreground">sạc điện thoại tương đương</p>
                </div>
              </motion.div>
              
              <motion.div 
                variants={fadeInUp}
                className="flex items-center gap-4 p-4 bg-background/80 rounded-xl border border-border transition-all hover:translate-x-2 hover:shadow-lg hover:shadow-emerald-500/20"
              >
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                  <Recycle className="w-8 h-8 text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-3xl font-bold text-foreground">
                    <AnimatedNumber value={equivalents.plasticBottles} /> <span className="text-lg font-medium">chai</span>
                  </p>
                  <p className="text-sm text-muted-foreground">nhựa sản xuất được bù trừ</p>
                </div>
              </motion.div>
            </motion.div>
          </motion.div>
        </div>

        {/* Action buttons */}
        <motion.div 
          variants={fadeInUp}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.8, delay: 3.2 }}
          className="flex flex-col sm:flex-row gap-4 mt-8 max-w-lg mx-auto lg:max-w-none"
        >
          <Button
            onClick={handleViewLeaderboard}
            className="flex-1 h-14 bg-gold hover:bg-gold/90 text-background font-semibold text-lg shadow-lg shadow-gold/30"
          >
            <Trophy className="w-6 h-6 mr-2" />
            Bảng xếp hạng
          </Button>
          
          <Button
            onClick={handlePlayAgain}
            variant="outline"
            className="flex-1 h-14 border-2 border-primary text-primary hover:bg-primary/10 font-semibold text-lg"
          >
            🌲 Đo cây khác
          </Button>
        </motion.div>
      </div>
    </div>
  );
};

export default ResultsScreen;
