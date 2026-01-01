import { useMemo } from 'react';

const FallingLeaves = () => {
  const leaves = useMemo(() => 
    [...Array(8)].map((_, i) => ({
      id: i,
      left: Math.random() * 100,
      delay: Math.random() * 5,
      duration: 10 + Math.random() * 8,
      size: 15 + Math.random() * 15,
    })), []
  );

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {leaves.map((leaf) => (
        <div
          key={leaf.id}
          className="absolute animate-falling-leaf will-change-transform"
          style={{
            left: `${leaf.left}%`,
            top: -50,
            width: leaf.size,
            height: leaf.size,
            animationDelay: `${leaf.delay}s`,
            animationDuration: `${leaf.duration}s`,
          }}
        >
          <svg viewBox="0 0 24 24" className="w-full h-full text-primary/50">
            <path
              fill="currentColor"
              d="M17,8C8,10 5.9,16.17 3.82,21.34L5.71,22L6.66,19.7C7.14,19.87 7.64,20 8,20C19,20 22,3 22,3C21,5 14,5.25 9,6.25C4,7.25 2,11.5 2,13.5C2,15.5 3.75,17.25 3.75,17.25C7,8 17,8 17,8Z"
            />
          </svg>
        </div>
      ))}
    </div>
  );
};

export default FallingLeaves;
