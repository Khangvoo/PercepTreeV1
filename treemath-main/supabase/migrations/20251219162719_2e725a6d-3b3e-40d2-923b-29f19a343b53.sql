-- Bảng dữ liệu chuẩn cho 17 cây
CREATE TABLE public.master_trees (
  id SERIAL PRIMARY KEY,
  tree_number INTEGER NOT NULL UNIQUE CHECK (tree_number >= 1 AND tree_number <= 17),
  actual_height DECIMAL(10, 2), -- Chiều cao thực tế (m)
  actual_diameter DECIMAL(10, 2), -- Đường kính thực tế (cm)
  species TEXT DEFAULT 'Chưa xác định',
  location_description TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Chèn 17 cây với dữ liệu mẫu (sẽ được cập nhật sau)
INSERT INTO public.master_trees (tree_number, actual_height, actual_diameter, species) VALUES
(1, 8.5, 35.0, 'Cây Bằng Lăng'),
(2, 10.2, 42.0, 'Cây Phượng'),
(3, 7.8, 28.0, 'Cây Xoài'),
(4, 12.0, 55.0, 'Cây Sấu'),
(5, 9.5, 38.0, 'Cây Bàng'),
(6, 11.0, 48.0, 'Cây Nhãn'),
(7, 6.5, 22.0, 'Cây Me'),
(8, 8.0, 32.0, 'Cây Dầu'),
(9, 14.5, 65.0, 'Cây Đa'),
(10, 7.2, 25.0, 'Cây Hoa Sữa'),
(11, 9.8, 40.0, 'Cây Muồng'),
(12, 10.5, 45.0, 'Cây Viết'),
(13, 8.8, 36.0, 'Cây Bạch Đàn'),
(14, 11.5, 52.0, 'Cây Keo'),
(15, 6.8, 24.0, 'Cây Trứng Cá'),
(16, 13.0, 58.0, 'Cây Xà Cừ'),
(17, 7.5, 30.0, 'Cây Lim');

-- Bảng lưu kết quả đo của người dùng
CREATE TABLE public.measurements (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tree_id INTEGER REFERENCES public.master_trees(id) ON DELETE CASCADE,
  user_name TEXT NOT NULL,
  user_class TEXT NOT NULL,
  measured_circumference DECIMAL(10, 2) NOT NULL, -- Chu vi đo được (cm)
  measured_angle DECIMAL(10, 2) NOT NULL, -- Góc đo (độ)
  measured_distance DECIMAL(10, 2) NOT NULL, -- Khoảng cách đứng (m)
  calculated_height DECIMAL(10, 2), -- Chiều cao tính được (m)
  calculated_diameter DECIMAL(10, 2), -- Đường kính tính được (cm)
  accuracy_score DECIMAL(5, 2), -- Độ chính xác (%)
  biomass_kg DECIMAL(10, 2), -- Sinh khối (kg)
  co2_absorbed_kg DECIMAL(10, 2), -- CO2 hấp thụ (kg)
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Bảng xếp hạng (derived từ measurements nhưng lưu riêng cho performance)
CREATE TABLE public.leaderboard (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  measurement_id UUID REFERENCES public.measurements(id) ON DELETE CASCADE,
  user_name TEXT NOT NULL,
  user_class TEXT NOT NULL,
  tree_number INTEGER NOT NULL,
  accuracy_score DECIMAL(5, 2) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.master_trees ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.measurements ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.leaderboard ENABLE ROW LEVEL SECURITY;

-- Policies cho master_trees (public read)
CREATE POLICY "Anyone can read master trees" ON public.master_trees
  FOR SELECT USING (true);

-- Policies cho measurements (public read/insert vì không cần auth)
CREATE POLICY "Anyone can insert measurements" ON public.measurements
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Anyone can read measurements" ON public.measurements
  FOR SELECT USING (true);

-- Policies cho leaderboard (public read/insert)
CREATE POLICY "Anyone can insert leaderboard" ON public.leaderboard
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Anyone can read leaderboard" ON public.leaderboard
  FOR SELECT USING (true);

-- Enable realtime cho leaderboard
ALTER PUBLICATION supabase_realtime ADD TABLE public.leaderboard;