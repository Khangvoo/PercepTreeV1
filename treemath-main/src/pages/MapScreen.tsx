import { useNavigate, useSearchParams } from 'react-router-dom';
import { useMeasurementStore } from '@/store/measurementStore';
import { supabase } from '@/integrations/supabase/client';
import { useEffect, useState, useMemo } from 'react';
import { TreeData } from '@/lib/calculations';

interface MasterTree {
  id: number;
  tree_number: number;
  actual_height: number;
  actual_diameter: number;
  species: string;
}

// Tree positions on map (x, y as percentages)
const treePositions: { [key: number]: { x: number; y: number } } = {
  1: { x: 15, y: 20 },
  2: { x: 30, y: 15 },
  3: { x: 50, y: 12 },
  4: { x: 70, y: 18 },
  5: { x: 85, y: 22 },
  6: { x: 20, y: 40 },
  7: { x: 40, y: 35 },
  8: { x: 60, y: 38 },
  9: { x: 80, y: 42 },
  10: { x: 25, y: 55 },
  11: { x: 45, y: 58 },
  12: { x: 65, y: 52 },
  13: { x: 15, y: 72 },
  14: { x: 35, y: 75 },
  15: { x: 55, y: 78 },
  16: { x: 75, y: 70 },
  17: { x: 88, y: 65 },
};

const MapScreen = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { setSelectedTree, selectedTreeId } = useMeasurementStore();
  const [trees, setTrees] = useState<MasterTree[]>([]);
  const [hoveredTree, setHoveredTree] = useState<number | null>(null);

  useEffect(() => {
    const fetchTrees = async () => {
      const { data } = await supabase
        .from('master_trees')
        .select('*')
        .order('tree_number');
      
      if (data) {
        setTrees(data);
        
        // Auto-select tree from URL parameter (from QR code)
        const treeParam = searchParams.get('tree');
        if (treeParam) {
          const treeNumber = parseInt(treeParam);
          const matchedTree = data.find(t => t.tree_number === treeNumber);
          if (matchedTree) {
            const treeData: TreeData = {
              treeNumber: matchedTree.tree_number,
              actualHeight: Number(matchedTree.actual_height),
              actualDiameter: Number(matchedTree.actual_diameter),
              species: matchedTree.species,
            };
            setSelectedTree(matchedTree.tree_number, treeData);
            // Navigate directly to user-info if coming from QR
            navigate('/user-info');
          }
        }
      }
    };
    fetchTrees();
  }, [searchParams, setSelectedTree, navigate]);

  const handleTreeSelect = (tree: MasterTree) => {
    const treeData: TreeData = {
      treeNumber: tree.tree_number,
      actualHeight: Number(tree.actual_height),
      actualDiameter: Number(tree.actual_diameter),
      species: tree.species,
    };
    setSelectedTree(tree.tree_number, treeData);
    navigate('/user-info');
  };

  // Memoize grid lines to prevent re-renders
  const gridLines = useMemo(() => (
    <svg className="absolute inset-0 w-full h-full opacity-20">
      {[...Array(10)].map((_, i) => (
        <line
          key={`h-${i}`}
          x1="0"
          y1={`${(i + 1) * 10}%`}
          x2="100%"
          y2={`${(i + 1) * 10}%`}
          stroke="currentColor"
          strokeWidth="0.5"
          className="text-primary"
        />
      ))}
      {[...Array(10)].map((_, i) => (
        <line
          key={`v-${i}`}
          x1={`${(i + 1) * 10}%`}
          y1="0"
          x2={`${(i + 1) * 10}%`}
          y2="100%"
          stroke="currentColor"
          strokeWidth="0.5"
          className="text-primary"
        />
      ))}
    </svg>
  ), []);

  return (
    <div className="min-h-screen bg-background p-4 flex flex-col">
      {/* Header */}
      <div className="text-center mb-6 animate-fade-in">
        <h1 className="text-2xl md:text-3xl font-bold text-foreground">
          BẢN ĐỒ CÂY XANH
        </h1>
        <p className="text-muted-foreground mt-2">
          Chọn một cây để bắt đầu đo
        </p>
      </div>

      {/* Map container */}
      <div className="flex-1 relative glass-card p-4 overflow-hidden animate-scale-in">
        {/* Map background with grid */}
        <div className="absolute inset-4 rounded-xl bg-gradient-to-br from-primary/10 to-background border border-primary/20">
          {/* Grid lines */}
          {gridLines}

          {/* Building placeholder */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 
                          w-32 h-20 bg-muted/50 rounded-lg border border-border flex items-center justify-center">
            <span className="text-xs text-muted-foreground">TRƯỜNG HỌC</span>
          </div>

          {/* Tree markers - using CSS transitions instead of Framer Motion */}
          {trees.map((tree) => {
            const pos = treePositions[tree.tree_number];
            if (!pos) return null;
            
            const isSelected = selectedTreeId === tree.tree_number;
            const isHovered = hoveredTree === tree.tree_number;
            
            return (
              <button
                key={tree.tree_number}
                className={`tree-marker absolute transition-transform duration-200 ${isSelected ? 'selected' : ''} ${isHovered ? 'scale-110' : ''}`}
                style={{
                  left: `${pos.x}%`,
                  top: `${pos.y}%`,
                  transform: `translate(-50%, -50%) ${isHovered ? 'scale(1.1)' : 'scale(1)'}`,
                }}
                onClick={() => handleTreeSelect(tree)}
                onMouseEnter={() => setHoveredTree(tree.tree_number)}
                onMouseLeave={() => setHoveredTree(null)}
              >
                <span className={`font-bold text-sm ${isSelected ? 'text-primary-foreground' : 'text-primary'}`}>
                  {tree.tree_number}
                </span>
              </button>
            );
          })}
        </div>

        {/* Tooltip for hovered tree */}
        {hoveredTree && (
          <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 
                         bg-card/90 backdrop-blur-sm px-4 py-2 rounded-lg border border-primary/30 animate-fade-in">
            <p className="text-sm text-foreground font-medium">
              Cây số {hoveredTree}: {trees.find(t => t.tree_number === hoveredTree)?.species}
            </p>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-center gap-6 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-primary/20 border-2 border-primary" />
          <span>Cây có thể đo</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-primary border-2 border-primary-foreground" />
          <span>Đã chọn</span>
        </div>
      </div>
    </div>
  );
};

export default MapScreen;
