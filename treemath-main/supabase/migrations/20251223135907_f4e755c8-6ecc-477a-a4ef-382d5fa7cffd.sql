-- Drop security_invoker views (they won't work with blocked base tables)
DROP VIEW IF EXISTS public.leaderboard_public;
DROP VIEW IF EXISTS public.measurements_public;

-- Create security definer functions to return masked data safely
-- These bypass RLS but only return masked data

CREATE OR REPLACE FUNCTION public.get_leaderboard_masked(p_tree_number INTEGER DEFAULT NULL, p_limit INTEGER DEFAULT 10)
RETURNS TABLE (
  id UUID,
  measurement_id UUID,
  tree_number INTEGER,
  accuracy_score NUMERIC,
  created_at TIMESTAMPTZ,
  user_name TEXT,
  user_class TEXT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    l.id,
    l.measurement_id,
    l.tree_number,
    l.accuracy_score,
    l.created_at,
    CASE
      WHEN l.user_name IS NULL OR btrim(l.user_name) = '' THEN 'Ẩn danh'
      ELSE left(btrim(l.user_name), 1) || '••'
    END AS user_name,
    CASE
      WHEN l.user_class IS NULL OR btrim(l.user_class) = '' THEN ''
      WHEN length(btrim(l.user_class)) >= 2 THEN left(btrim(l.user_class), 2) || '•'
      ELSE left(btrim(l.user_class), 1) || '••'
    END AS user_class
  FROM public.leaderboard l
  WHERE (p_tree_number IS NULL OR l.tree_number = p_tree_number)
  ORDER BY l.accuracy_score DESC
  LIMIT p_limit;
$$;

CREATE OR REPLACE FUNCTION public.get_measurements_masked(p_tree_id INTEGER DEFAULT NULL, p_limit INTEGER DEFAULT 50)
RETURNS TABLE (
  id UUID,
  tree_id INTEGER,
  measured_circumference NUMERIC,
  measured_height NUMERIC,
  calculated_height NUMERIC,
  calculated_diameter NUMERIC,
  accuracy_score NUMERIC,
  biomass_kg NUMERIC,
  co2_absorbed_kg NUMERIC,
  created_at TIMESTAMPTZ,
  user_name TEXT,
  user_class TEXT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    m.id,
    m.tree_id,
    m.measured_circumference,
    m.measured_height,
    m.calculated_height,
    m.calculated_diameter,
    m.accuracy_score,
    m.biomass_kg,
    m.co2_absorbed_kg,
    m.created_at,
    CASE
      WHEN m.user_name IS NULL OR btrim(m.user_name) = '' THEN 'Ẩn danh'
      ELSE left(btrim(m.user_name), 1) || '••'
    END AS user_name,
    CASE
      WHEN m.user_class IS NULL OR btrim(m.user_class) = '' THEN ''
      WHEN length(btrim(m.user_class)) >= 2 THEN left(btrim(m.user_class), 2) || '•'
      ELSE left(btrim(m.user_class), 1) || '••'
    END AS user_class
  FROM public.measurements m
  WHERE (p_tree_id IS NULL OR m.tree_id = p_tree_id)
  ORDER BY m.created_at DESC
  LIMIT p_limit;
$$;

-- Grant execute to public roles
GRANT EXECUTE ON FUNCTION public.get_leaderboard_masked(INTEGER, INTEGER) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_measurements_masked(INTEGER, INTEGER) TO anon, authenticated;