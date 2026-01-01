-- Add measured_height column to measurements table
ALTER TABLE public.measurements 
ADD COLUMN measured_height numeric NULL;

-- Drop old angle and distance columns (they are no longer used)
ALTER TABLE public.measurements 
DROP COLUMN IF EXISTS measured_angle;

ALTER TABLE public.measurements 
DROP COLUMN IF EXISTS measured_distance;