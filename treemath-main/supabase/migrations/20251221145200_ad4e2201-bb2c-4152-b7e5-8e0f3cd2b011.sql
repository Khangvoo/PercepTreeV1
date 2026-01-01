-- Add explicit DENY policies for UPDATE and DELETE on measurements table
CREATE POLICY "Deny all updates to measurements" 
  ON public.measurements
  FOR UPDATE 
  USING (false);

CREATE POLICY "Deny all deletes to measurements" 
  ON public.measurements
  FOR DELETE 
  USING (false);

-- Add explicit DENY policies for UPDATE and DELETE on leaderboard table
CREATE POLICY "Deny all updates to leaderboard" 
  ON public.leaderboard
  FOR UPDATE 
  USING (false);

CREATE POLICY "Deny all deletes to leaderboard" 
  ON public.leaderboard
  FOR DELETE 
  USING (false);