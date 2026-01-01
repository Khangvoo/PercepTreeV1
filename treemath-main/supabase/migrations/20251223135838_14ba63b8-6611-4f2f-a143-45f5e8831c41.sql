-- Fix SECURITY DEFINER view warnings by adding security_invoker option

-- Recreate views with SECURITY INVOKER (not DEFINER)
DROP VIEW IF EXISTS public.leaderboard_public;
CREATE VIEW public.leaderboard_public
WITH (security_invoker = true)
AS
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

DROP VIEW IF EXISTS public.measurements_public;
CREATE VIEW public.measurements_public
WITH (security_invoker = true)
AS
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