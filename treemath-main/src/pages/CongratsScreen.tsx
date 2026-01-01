import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useMeasurementStore } from '@/store/measurementStore';
import { Button } from '@/components/ui/button';
import { useEffect } from 'react';
import { Sparkles, ArrowRight } from 'lucide-react';
import confetti from 'canvas-confetti';

const CongratsScreen = () => {
  const navigate = useNavigate();
  const { result, userName } = useMeasurementStore();

  useEffect(() => {
    // Trigger confetti
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 },
      colors: ['#22c55e', '#16a34a', '#15803d', '#86efac'],
    });
  }, []);

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

  const displayUserName = userName || 'Học sinh';

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 90) return 'text-gold';
    if (accuracy >= 70) return 'text-primary';
    if (accuracy >= 50) return 'text-accent';
    return 'text-muted-foreground';
  };

  const getAccuracyMessage = (accuracy: number) => {
    if (accuracy >= 95) return 'Xuất sắc! 🌟';
    if (accuracy >= 85) return 'Tuyệt vời! 🎉';
    if (accuracy >= 70) return 'Khá tốt! 👍';
    if (accuracy >= 50) return 'Cần cải thiện! 💪';
    return 'Hãy thử lại! 🔄';
  };

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* Background glow */}
      <div className="absolute inset-0 bg-gradient-radial from-primary/10 via-transparent to-transparent" />

      <motion.div
        className="text-center z-10"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* Star icon */}
        <motion.div
          className="w-24 h-24 mx-auto mb-6 rounded-full bg-gold/20 
                     flex items-center justify-center border-2 border-gold/50"
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 5, -5, 0],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
          }}
        >
          <Sparkles className="w-12 h-12 text-gold" />
        </motion.div>

        {/* Congratulations text */}
        <motion.h1
          className="text-3xl md:text-4xl font-bold text-foreground mb-2"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Chúc mừng{displayUserName ? `, ${displayUserName}` : ''}!
        </motion.h1>

        <motion.p
          className="text-lg text-muted-foreground mb-8"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          {getAccuracyMessage(displayResult.overallAccuracy)}
        </motion.p>

        {/* Accuracy score */}
        <motion.div
          className="glass-card p-8 mb-8"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <p className="text-muted-foreground mb-2">Độ chính xác của bạn</p>
          <motion.p
            className={`text-6xl md:text-7xl font-bold ${getAccuracyColor(displayResult.overallAccuracy)}`}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.6, type: "spring", stiffness: 200 }}
          >
            {displayResult.overallAccuracy}%
          </motion.p>
          
          <div className="flex justify-center gap-8 mt-6 text-sm">
            <div>
              <p className="text-muted-foreground">Chiều cao</p>
              <p className="text-foreground font-semibold">{displayResult.heightAccuracy}%</p>
            </div>
            <div>
              <p className="text-muted-foreground">Đường kính</p>
              <p className="text-foreground font-semibold">{displayResult.diameterAccuracy}%</p>
            </div>
          </div>
        </motion.div>

        {/* Continue button */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <Button
            onClick={() => navigate('/results')}
            className="h-12 px-8 text-lg font-semibold bg-primary hover:bg-primary/90 
                       text-primary-foreground glow-primary"
          >
            Xem kết quả chi tiết
            <ArrowRight className="w-5 h-5 ml-2" />
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default CongratsScreen;
