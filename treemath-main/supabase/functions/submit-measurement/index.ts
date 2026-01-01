import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Rate limit: max 10 requests per minute per IP
const RATE_LIMIT_WINDOW_MS = 60 * 1000;
const MAX_REQUESTS_PER_WINDOW = 10;

// Input validation
function validateInput(data: unknown): { valid: boolean; error?: string; parsed?: Record<string, unknown> } {
  if (!data || typeof data !== 'object') {
    return { valid: false, error: 'Invalid request body' };
  }

  const body = data as Record<string, unknown>;

  // Required fields
  const requiredFields = ['tree_id', 'user_name', 'user_class', 'measured_circumference', 'measured_height', 'calculated_height', 'calculated_diameter', 'accuracy_score', 'biomass_kg', 'co2_absorbed_kg'];
  for (const field of requiredFields) {
    if (body[field] === undefined || body[field] === null) {
      return { valid: false, error: `Missing required field: ${field}` };
    }
  }

  // Validate types and ranges
  const treeId = Number(body.tree_id);
  if (!Number.isInteger(treeId) || treeId < 1 || treeId > 100) {
    return { valid: false, error: 'Invalid tree_id: must be integer 1-100' };
  }

  const userName = String(body.user_name).trim().slice(0, 50);
  if (userName.length < 1 || userName.length > 50) {
    return { valid: false, error: 'Invalid user_name: must be 1-50 characters' };
  }
  // Allow Unicode letters, numbers, spaces, hyphens, apostrophes
  if (!/^[\p{L}\p{N}\s\-''.]+$/u.test(userName)) {
    return { valid: false, error: 'Invalid user_name: contains invalid characters' };
  }

  const userClass = String(body.user_class).trim().slice(0, 10);
  if (userClass.length < 1 || userClass.length > 10) {
    return { valid: false, error: 'Invalid user_class: must be 1-10 characters' };
  }
  // Allow alphanumeric class identifiers like "10A1", "11B2"
  if (!/^[0-9]{1,2}[A-Za-z][0-9]{0,2}$/i.test(userClass)) {
    return { valid: false, error: 'Invalid user_class format: expected format like 10A1' };
  }

  // Numeric validations
  const numericFields = [
    { name: 'measured_circumference', min: 1, max: 1000 },
    { name: 'measured_height', min: 0.1, max: 100 },
    { name: 'calculated_height', min: 0.1, max: 200 },
    { name: 'calculated_diameter', min: 0.1, max: 500 },
    { name: 'accuracy_score', min: 0, max: 100 },
    { name: 'biomass_kg', min: 0, max: 100000 },
    { name: 'co2_absorbed_kg', min: 0, max: 200000 },
  ];

  const parsed: Record<string, unknown> = {
    tree_id: treeId,
    user_name: userName,
    user_class: userClass,
  };

  for (const field of numericFields) {
    const value = Number(body[field.name]);
    if (isNaN(value) || value < field.min || value > field.max) {
      return { valid: false, error: `Invalid ${field.name}: must be between ${field.min} and ${field.max}` };
    }
    parsed[field.name] = Math.round(value * 100) / 100; // Round to 2 decimal places
  }

  return { valid: true, parsed };
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  const startTime = Date.now();
  console.log(`[submit-measurement] Request started at ${new Date().toISOString()}`);

  try {
    if (req.method !== 'POST') {
      console.log(`[submit-measurement] Invalid method: ${req.method}`);
      return new Response(
        JSON.stringify({ error: 'Method not allowed' }),
        { status: 405, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Get client IP for rate limiting
    const clientIp = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 
                     req.headers.get('x-real-ip') || 
                     'unknown';
    console.log(`[submit-measurement] Client IP: ${clientIp}`);

    // Parse request body
    let body: unknown;
    try {
      body = await req.json();
    } catch {
      console.log('[submit-measurement] Failed to parse request body');
      return new Response(
        JSON.stringify({ error: 'Invalid JSON body' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Validate input
    const validation = validateInput(body);
    if (!validation.valid) {
      console.log(`[submit-measurement] Validation failed: ${validation.error}`);
      return new Response(
        JSON.stringify({ error: validation.error }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const parsed = validation.parsed!;
    console.log(`[submit-measurement] Validated input for tree ${parsed.tree_id}, user: ${parsed.user_name}`);

    // Initialize Supabase client with service role (bypasses RLS)
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Check IP-based rate limit
    const windowStart = new Date(Math.floor(Date.now() / RATE_LIMIT_WINDOW_MS) * RATE_LIMIT_WINDOW_MS).toISOString();
    
    const { data: rateLimitData, error: rateLimitFetchError } = await supabase
      .from('request_rate_limits')
      .select('count')
      .eq('ip', clientIp)
      .eq('action', 'submit_measurement')
      .eq('window_start', windowStart)
      .single();

    if (rateLimitFetchError && rateLimitFetchError.code !== 'PGRST116') {
      // PGRST116 = no rows found, which is fine
      console.error('[submit-measurement] Rate limit fetch error:', rateLimitFetchError);
    }

    const currentCount = rateLimitData?.count || 0;
    console.log(`[submit-measurement] Current rate limit count for IP ${clientIp}: ${currentCount}/${MAX_REQUESTS_PER_WINDOW}`);

    if (currentCount >= MAX_REQUESTS_PER_WINDOW) {
      console.log(`[submit-measurement] Rate limit exceeded for IP ${clientIp}`);
      return new Response(
        JSON.stringify({ error: 'Rate limit exceeded. Please try again later.' }),
        { status: 429, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Update rate limit counter
    const { error: upsertError } = await supabase
      .from('request_rate_limits')
      .upsert({
        ip: clientIp,
        action: 'submit_measurement',
        window_start: windowStart,
        count: currentCount + 1,
      }, { onConflict: 'ip,action,window_start' });

    if (upsertError) {
      console.error('[submit-measurement] Failed to update rate limit:', upsertError);
      // Continue anyway - don't block legitimate requests due to rate limit tracking issues
    }

    // Verify tree exists
    const { data: treeData, error: treeError } = await supabase
      .from('master_trees')
      .select('tree_number')
      .eq('tree_number', parsed.tree_id)
      .single();

    if (treeError || !treeData) {
      console.log(`[submit-measurement] Tree not found: ${parsed.tree_id}`);
      return new Response(
        JSON.stringify({ error: 'Invalid tree reference' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Insert measurement
    const { data: measurementData, error: measurementError } = await supabase
      .from('measurements')
      .insert({
        tree_id: parsed.tree_id,
        user_name: parsed.user_name,
        user_class: parsed.user_class,
        measured_circumference: parsed.measured_circumference,
        measured_height: parsed.measured_height,
        calculated_height: parsed.calculated_height,
        calculated_diameter: parsed.calculated_diameter,
        accuracy_score: parsed.accuracy_score,
        biomass_kg: parsed.biomass_kg,
        co2_absorbed_kg: parsed.co2_absorbed_kg,
      })
      .select()
      .single();

    if (measurementError) {
      console.error('[submit-measurement] Measurement insert error:', measurementError);
      return new Response(
        JSON.stringify({ error: 'Failed to save measurement' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log(`[submit-measurement] Measurement saved: ${measurementData.id}`);

    // Insert leaderboard entry
    const { error: leaderboardError } = await supabase
      .from('leaderboard')
      .insert({
        measurement_id: measurementData.id,
        user_name: parsed.user_name,
        user_class: parsed.user_class,
        tree_number: parsed.tree_id,
        accuracy_score: parsed.accuracy_score,
      });

    if (leaderboardError) {
      console.error('[submit-measurement] Leaderboard insert error:', leaderboardError);
      // Don't fail the whole request - measurement was saved successfully
    } else {
      console.log('[submit-measurement] Leaderboard entry saved');
    }

    const duration = Date.now() - startTime;
    console.log(`[submit-measurement] Request completed in ${duration}ms`);

    return new Response(
      JSON.stringify({ 
        success: true, 
        measurement_id: measurementData.id 
      }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('[submit-measurement] Unexpected error:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
