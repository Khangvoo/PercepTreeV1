-- Lock down public write access: only backend functions (service role) should insert

-- measurements: remove permissive insert policy
DROP POLICY IF EXISTS "Anyone can insert measurements" ON public.measurements;
CREATE POLICY "Deny all inserts to measurements"
ON public.measurements
FOR INSERT
WITH CHECK (false);

-- leaderboard: remove permissive insert policy
DROP POLICY IF EXISTS "Anyone can insert leaderboard" ON public.leaderboard;
CREATE POLICY "Deny all inserts to leaderboard"
ON public.leaderboard
FOR INSERT
WITH CHECK (false);

-- Add an internal rate-limit ledger for backend functions (service-role writes only)
CREATE TABLE IF NOT EXISTS public.request_rate_limits (
  ip TEXT NOT NULL,
  action TEXT NOT NULL,
  window_start TIMESTAMPTZ NOT NULL,
  count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ip, action, window_start)
);

ALTER TABLE public.request_rate_limits ENABLE ROW LEVEL SECURITY;

-- No RLS policies on request_rate_limits: default-deny for anon/authenticated.
-- Service role bypasses RLS and can read/write for enforcement.

-- Add global circuit-breaker rate limits at the database layer (protects against distributed floods)
CREATE OR REPLACE FUNCTION public.check_global_measurements_rate_limit()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  recent_global INTEGER;
BEGIN
  SELECT COUNT(*) INTO recent_global
  FROM public.measurements
  WHERE created_at > now() - interval '1 minute';

  IF recent_global >= 200 THEN
    RAISE EXCEPTION 'System rate limit exceeded. Try again later.'
      USING ERRCODE = 'P0001';
  END IF;

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS enforce_global_measurements_rate_limit ON public.measurements;
CREATE TRIGGER enforce_global_measurements_rate_limit
  BEFORE INSERT ON public.measurements
  FOR EACH ROW
  EXECUTE FUNCTION public.check_global_measurements_rate_limit();

CREATE OR REPLACE FUNCTION public.check_global_leaderboard_rate_limit()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  recent_global INTEGER;
BEGIN
  SELECT COUNT(*) INTO recent_global
  FROM public.leaderboard
  WHERE created_at > now() - interval '1 minute';

  IF recent_global >= 500 THEN
    RAISE EXCEPTION 'System rate limit exceeded. Try again later.'
      USING ERRCODE = 'P0001';
  END IF;

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS enforce_global_leaderboard_rate_limit ON public.leaderboard;
CREATE TRIGGER enforce_global_leaderboard_rate_limit
  BEFORE INSERT ON public.leaderboard
  FOR EACH ROW
  EXECUTE FUNCTION public.check_global_leaderboard_rate_limit();
