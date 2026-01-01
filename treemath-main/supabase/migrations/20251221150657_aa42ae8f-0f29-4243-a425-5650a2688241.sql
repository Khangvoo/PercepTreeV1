-- Add SELECT policy to measurements table to allow public read access
-- This matches the existing pattern used by the leaderboard table
CREATE POLICY "Anyone can read measurements" 
  ON public.measurements 
  FOR SELECT 
  USING (true);