import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useMeasurementStore } from '@/store/measurementStore';
import { calculateTreeMeasurement } from '@/lib/calculations';
import { useEffect, useState } from 'react';

const LoadingScreen = () => {
  const navigate = useNavigate();
  const { selectedTree, circumference, height, setResult } = useMeasurementStore();
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState('Đang tính toán...');

  // Mock data for development preview
  const displayTree = selectedTree || {
    treeNumber: 1,
    species: 'Sao đen',
    actualHeight: 14,
    actualDiameter: 38,
  };

  useEffect(() => {
    // Simulate calculation steps
    const steps = [
      { progress: 25, text: 'Tính đường kính...' },
      { progress: 50, text: 'So sánh dữ liệu...' },
      { progress: 75, text: 'Tính sinh khối...' },
      { progress: 100, text: 'Hoàn thành!' },
    ];

    let stepIndex = 0;
    const interval = setInterval(() => {
      if (stepIndex < steps.length) {
        setProgress(steps[stepIndex].progress);
        setStatusText(steps[stepIndex].text);
        stepIndex++;
      } else {
        clearInterval(interval);
        
        // Only calculate and navigate if we have real data
        if (selectedTree) {
          const result = calculateTreeMeasurement(
            { circumference, height },
            selectedTree
          );
          setResult(result);
          
          // Navigate to result
          setTimeout(() => navigate('/congrats'), 500);
        }
        // If no selectedTree (dev mode), just stay on the page
      }
    }, 600);

    return () => clearInterval(interval);
  }, [selectedTree, circumference, height, setResult, navigate]);

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      {/* Tree filling animation */}
      <div className="relative w-64 h-80 mb-8">
        {/* Tree outline (gray) */}
        <svg viewBox="0 0 200 300" className="absolute inset-0 w-full h-full">
          {/* Trunk */}
          <rect
            x="85"
            y="200"
            width="30"
            height="80"
            rx="5"
            className="fill-muted/30"
          />
          {/* Crown layers */}
          <ellipse cx="100" cy="180" rx="60" ry="40" className="fill-muted/20" />
          <ellipse cx="100" cy="140" rx="50" ry="35" className="fill-muted/20" />
          <ellipse cx="100" cy="105" rx="40" ry="30" className="fill-muted/20" />
          <ellipse cx="100" cy="75" rx="25" ry="20" className="fill-muted/20" />
        </svg>

        {/* Tree filled (green) - clips based on progress */}
        <motion.div
          className="absolute inset-0"
          style={{
            clipPath: `inset(${100 - progress}% 0 0 0)`,
          }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <svg viewBox="0 0 200 300" className="w-full h-full">
            {/* Trunk */}
            <rect
              x="85"
              y="200"
              width="30"
              height="80"
              rx="5"
              className="fill-amber-700"
            />
            {/* Crown layers */}
            <ellipse cx="100" cy="180" rx="60" ry="40" className="fill-tree-green-dark" />
            <ellipse cx="100" cy="140" rx="50" ry="35" className="fill-primary" />
            <ellipse cx="100" cy="105" rx="40" ry="30" className="fill-tree-green-light" />
            <ellipse cx="100" cy="75" rx="25" ry="20" className="fill-primary" />
          </svg>
        </motion.div>

        {/* Progress percentage */}
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <span className="text-4xl font-bold text-foreground text-glow">
            {progress}%
          </span>
        </motion.div>
      </div>

      {/* Status text */}
      <motion.p
        key={statusText}
        className="text-lg text-muted-foreground"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        {statusText}
      </motion.p>

      {/* Progress bar */}
      <div className="w-64 h-2 bg-muted/30 rounded-full mt-6 overflow-hidden">
        <motion.div
          className="h-full bg-primary rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>
    </div>
  );
};

export default LoadingScreen;
