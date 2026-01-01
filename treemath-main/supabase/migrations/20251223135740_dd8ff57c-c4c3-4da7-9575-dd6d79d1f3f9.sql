-- Fix PUBLIC_DATA_EXPOSURE (warn): mask student identifiers for public reads

-- 1) Create masked public views
CREATE OR REPLACE VIEW public.leaderboard_public AS
SELECT
  id,
  measurement_id,
  tree_number,
  accuracy_score,
  created_at,
  CASE
    WHEN user_name IS NULL OR btrim(user_name) = '' THEN 'Ẩn danh'
    ELSE left(btrim(user_name), 1) || '••'
  END AS user_name,
  CASE
    WHEN user_class IS NULL OR btrim(user_class) = '' THEN ''
    WHEN length(btrim(user_class)) >= 2 THEN left(btrim(user_class), 2) || '•'
    ELSE left(btrim(user_class), 1) || '••'
  END AS user_class
FROM public.leaderboard;

CREATE OR REPLACE VIEW public.measurements_public AS
SELECT
  id,
  tree_id,
  measured_circumference,
  measured_height,
  calculated_height,
  calculated_diameter,
  accuracy_score,
  biomass_kg,
  co2_absorbed_kg,
  created_at,
  CASE
    WHEN user_name IS NULL OR btrim(user_name) = '' THEN 'Ẩn danh'
    ELSE left(btrim(user_name), 1) || '••'
  END AS user_name,
  CASE
    WHEN user_class IS NULL OR btrim(user_class) = '' THEN ''
    WHEN length(btrim(user_class)) >= 2 THEN left(btrim(user_class), 2) || '•'
    ELSE left(btrim(user_class), 1) || '••'
  END AS user_class
FROM public.measurements;

-- Ensure public roles can read the masked views
GRANT SELECT ON public.leaderboard_public TO anon, authenticated;
GRANT SELECT ON public.measurements_public TO anon, authenticated;

-- 2) Block direct public reads from base tables (writes remain through backend function)
DROP POLICY IF EXISTS "Anyone can read leaderboard" ON public.leaderboard;
CREATE POLICY "Deny all selects to leaderboard"
ON public.leaderboard
FOR SELECT
USING (false);

DROP POLICY IF EXISTS "Anyone can read measurements" ON public.measurements;
CREATE POLICY "Deny all selects to measurements"
ON public.measurements
FOR SELECT
USING (false);
