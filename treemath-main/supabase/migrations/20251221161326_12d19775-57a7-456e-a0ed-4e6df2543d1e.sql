-- Fix the linter warning: add explicit deny policies for request_rate_limits (service role bypasses RLS, so these only protect anon/authenticated)
CREATE POLICY "Deny all select to request_rate_limits"
ON public.request_rate_limits
FOR SELECT
USING (false);

CREATE POLICY "Deny all inserts to request_rate_limits"
ON public.request_rate_limits
FOR INSERT
WITH CHECK (false);

CREATE POLICY "Deny all updates to request_rate_limits"
ON public.request_rate_limits
FOR UPDATE
USING (false);

CREATE POLICY "Deny all deletes to request_rate_limits"
ON public.request_rate_limits
FOR DELETE
USING (false);