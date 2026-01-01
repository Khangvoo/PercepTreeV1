export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.1"
  }
  public: {
    Tables: {
      leaderboard: {
        Row: {
          accuracy_score: number
          created_at: string | null
          id: string
          measurement_id: string | null
          tree_number: number
          user_class: string
          user_name: string
        }
        Insert: {
          accuracy_score: number
          created_at?: string | null
          id?: string
          measurement_id?: string | null
          tree_number: number
          user_class: string
          user_name: string
        }
        Update: {
          accuracy_score?: number
          created_at?: string | null
          id?: string
          measurement_id?: string | null
          tree_number?: number
          user_class?: string
          user_name?: string
        }
        Relationships: [
          {
            foreignKeyName: "leaderboard_measurement_id_fkey"
            columns: ["measurement_id"]
            isOneToOne: false
            referencedRelation: "measurements"
            referencedColumns: ["id"]
          },
        ]
      }
      master_trees: {
        Row: {
          actual_diameter: number | null
          actual_height: number | null
          created_at: string | null
          id: number
          location_description: string | null
          species: string | null
          tree_number: number
          updated_at: string | null
        }
        Insert: {
          actual_diameter?: number | null
          actual_height?: number | null
          created_at?: string | null
          id?: number
          location_description?: string | null
          species?: string | null
          tree_number: number
          updated_at?: string | null
        }
        Update: {
          actual_diameter?: number | null
          actual_height?: number | null
          created_at?: string | null
          id?: number
          location_description?: string | null
          species?: string | null
          tree_number?: number
          updated_at?: string | null
        }
        Relationships: []
      }
      measurements: {
        Row: {
          accuracy_score: number | null
          biomass_kg: number | null
          calculated_diameter: number | null
          calculated_height: number | null
          co2_absorbed_kg: number | null
          created_at: string | null
          id: string
          measured_circumference: number
          measured_height: number | null
          tree_id: number | null
          user_class: string
          user_name: string
        }
        Insert: {
          accuracy_score?: number | null
          biomass_kg?: number | null
          calculated_diameter?: number | null
          calculated_height?: number | null
          co2_absorbed_kg?: number | null
          created_at?: string | null
          id?: string
          measured_circumference: number
          measured_height?: number | null
          tree_id?: number | null
          user_class: string
          user_name: string
        }
        Update: {
          accuracy_score?: number | null
          biomass_kg?: number | null
          calculated_diameter?: number | null
          calculated_height?: number | null
          co2_absorbed_kg?: number | null
          created_at?: string | null
          id?: string
          measured_circumference?: number
          measured_height?: number | null
          tree_id?: number | null
          user_class?: string
          user_name?: string
        }
        Relationships: [
          {
            foreignKeyName: "measurements_tree_id_fkey"
            columns: ["tree_id"]
            isOneToOne: false
            referencedRelation: "master_trees"
            referencedColumns: ["id"]
          },
        ]
      }
      request_rate_limits: {
        Row: {
          action: string
          count: number
          created_at: string
          ip: string
          window_start: string
        }
        Insert: {
          action: string
          count?: number
          created_at?: string
          ip: string
          window_start: string
        }
        Update: {
          action?: string
          count?: number
          created_at?: string
          ip?: string
          window_start?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      get_leaderboard_masked: {
        Args: { p_limit?: number; p_tree_number?: number }
        Returns: {
          accuracy_score: number
          created_at: string
          id: string
          measurement_id: string
          tree_number: number
          user_class: string
          user_name: string
        }[]
      }
      get_measurements_masked: {
        Args: { p_limit?: number; p_tree_id?: number }
        Returns: {
          accuracy_score: number
          biomass_kg: number
          calculated_diameter: number
          calculated_height: number
          co2_absorbed_kg: number
          created_at: string
          id: string
          measured_circumference: number
          measured_height: number
          tree_id: number
          user_class: string
          user_name: string
        }[]
      }
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
