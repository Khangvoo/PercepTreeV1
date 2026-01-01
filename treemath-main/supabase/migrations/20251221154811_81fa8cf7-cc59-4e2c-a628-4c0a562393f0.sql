-- Fix tree_id validation: Add trigger to validate tree references before insert
-- Using a trigger instead of altering foreign key to avoid breaking existing data

CREATE OR REPLACE FUNCTION validate_tree_id()
RETURNS TRIGGER AS $$
BEGIN
  -- Check if tree_id is provided and valid
  IF NEW.tree_id IS NOT NULL THEN
    -- Verify tree_number exists in master_trees
    IF NOT EXISTS (
      SELECT 1 FROM public.master_trees 
      WHERE tree_number = NEW.tree_id
    ) THEN
      RAISE EXCEPTION 'Invalid tree reference: tree_number % does not exist', NEW.tree_id
        USING ERRCODE = 'P0001';
    END IF;
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY INVOKER SET search_path = public;

-- Create trigger to enforce tree_id validation on measurements insert
CREATE TRIGGER validate_measurement_tree_id
  BEFORE INSERT ON public.measurements
  FOR EACH ROW
  EXECUTE FUNCTION validate_tree_id();