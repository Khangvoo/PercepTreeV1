import { create } from 'zustand';
import { TreeData, CalculationResult } from '@/lib/calculations';

interface MeasurementState {
  // Selected tree
  selectedTreeId: number | null;
  selectedTree: TreeData | null;
  
  // User info
  userName: string;
  userClass: string;
  
  // Measurement data
  circumference: number;
  height: number;
  
  // Results
  result: CalculationResult | null;
  
  // Actions
  setSelectedTree: (id: number, tree: TreeData) => void;
  setUserInfo: (name: string, className: string) => void;
  setMeasurements: (circumference: number, height: number) => void;
  setResult: (result: CalculationResult) => void;
  reset: () => void;
}

export const useMeasurementStore = create<MeasurementState>((set) => ({
  selectedTreeId: null,
  selectedTree: null,
  userName: '',
  userClass: '',
  circumference: 0,
  height: 0,
  result: null,
  
  setSelectedTree: (id, tree) => set({ selectedTreeId: id, selectedTree: tree }),
  
  setUserInfo: (name, className) => set({ userName: name, userClass: className }),
  
  setMeasurements: (circumference, height) => 
    set({ circumference, height }),
  
  setResult: (result) => set({ result }),
  
  reset: () => set({
    selectedTreeId: null,
    selectedTree: null,
    userName: '',
    userClass: '',
    circumference: 0,
    height: 0,
    result: null,
  }),
}));
