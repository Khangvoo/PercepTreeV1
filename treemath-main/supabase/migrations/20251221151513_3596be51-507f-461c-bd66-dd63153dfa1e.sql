-- Fix 1: Add explicit DENY policies for UPDATE and DELETE on master_trees table
-- This prevents unauthorized modification of reference tree data

CREATE POLICY "Deny all updates to master_trees" 
  ON public.master_trees
  FOR UPDATE 
  USING (false);

CREATE POLICY "Deny all deletes to master_trees" 
  ON public.master_trees
  FOR DELETE 
  USING (false);

-- Also add INSERT deny policy since this is reference data that should only be managed by admins
CREATE POLICY "Deny all inserts to master_trees" 
  ON public.master_trees
  FOR INSERT 
  WITH CHECK (false);

-- Fix 2: Add server-side rate limiting for measurements table
-- This enforces rate limiting at the database level (max 5 submissions per user_name+user_class per minute)

CREATE OR REPLACE FUNCTION check_measurement_rate_limit()
RETURNS TRIGGER AS $$
DECLARE
  recent_count INTEGER;
BEGIN
  -- Count submissions from the same user in the last minute
  SELECT COUNT(*) INTO recent_count
  FROM public.measurements
  WHERE user_name = NEW.user_name
    AND user_class = NEW.user_class
    AND created_at > now() - interval '1 minute';
  
  -- Block if rate limit exceeded (5 per minute)
  IF recent_count >= 5 THEN
    RAISE EXCEPTION 'Rate limit exceeded: maximum 5 submissions per minute per user'
      USING ERRCODE = 'P0001';
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

-- Create trigger to enforce rate limit on insert
CREATE TRIGGER enforce_measurement_rate_limit
  BEFORE INSERT ON public.measurements
  FOR EACH ROW
  EXECUTE FUNCTION check_measurement_rate_limit();

-- Also add rate limiting for leaderboard table
CREATE OR REPLACE FUNCTION check_leaderboard_rate_limit()
RETURNS TRIGGER AS $$
DECLARE
  recent_count INTEGER;
BEGIN
  -- Count submissions from the same user in the last minute
  SELECT COUNT(*) INTO recent_count
  FROM public.leaderboard
  WHERE user_name = NEW.user_name
    AND user_class = NEW.user_class
    AND created_at > now() - interval '1 minute';
  
  -- Block if rate limit exceeded (5 per minute)
  IF recent_count >= 5 THEN
    RAISE EXCEPTION 'Rate limit exceeded: maximum 5 submissions per minute per user'
      USING ERRCODE = 'P0001';
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

-- Create trigger to enforce rate limit on insert
CREATE TRIGGER enforce_leaderboard_rate_limit
  BEFORE INSERT ON public.leaderboard
  FOR EACH ROW
  EXECUTE FUNCTION check_leaderboard_rate_limit();