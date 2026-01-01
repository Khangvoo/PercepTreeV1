import { useEffect, useState, useRef } from "react";
import { QRCodeSVG } from "qrcode.react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Printer, Download, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

interface MasterTree {
  id: number;
  tree_number: number;
  species: string;
  location_description: string | null;
}

const QRCodesScreen = () => {
  const navigate = useNavigate();
  const [trees, setTrees] = useState<MasterTree[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const printRef = useRef<HTMLDivElement>(null);

  // Get base URL for QR codes
  const baseUrl = window.location.origin;

  useEffect(() => {
    const fetchTrees = async () => {
      setIsLoading(true);
      const { data, error } = await supabase
        .from("master_trees")
        .select("id, tree_number, species, location_description")
        .order("tree_number");

      if (!error && data) {
        setTrees(data);
      }
      setIsLoading(false);
    };

    fetchTrees();
  }, []);

  const handlePrint = () => {
    window.print();
  };

  const downloadSingleQR = (treeNumber: number) => {
    const svg = document.getElementById(`qr-${treeNumber}`);
    if (!svg) return;

    const svgData = new XMLSerializer().serializeToString(svg);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx?.drawImage(img, 0, 0);
      const pngFile = canvas.toDataURL("image/png");
      const downloadLink = document.createElement("a");
      downloadLink.download = `tree-${treeNumber}-qr.png`;
      downloadLink.href = pngFile;
      downloadLink.click();
    };

    img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData)));
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header - Hidden when printing */}
      <div className="print:hidden sticky top-0 z-10 bg-background/95 backdrop-blur-sm border-b border-border p-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" onClick={() => navigate("/")}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <h1 className="text-xl font-bold text-foreground">QR Codes - Tree-Math</h1>
              <p className="text-sm text-muted-foreground">
                {trees.length} cây • In ra và dán lên bảng thông tin
              </p>
            </div>
          </div>
          <Button onClick={handlePrint} className="gap-2">
            <Printer className="w-4 h-4" />
            In tất cả
          </Button>
        </div>
      </div>

      {/* QR Grid */}
      <div ref={printRef} className="max-w-6xl mx-auto p-4 md:p-8">
        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">
            Đang tải danh sách cây...
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 print:grid-cols-2 print:gap-4">
            {trees.map((tree, index) => (
              <motion.div
                key={tree.tree_number}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="bg-card border border-border rounded-xl p-6 flex flex-col items-center print:break-inside-avoid print:border-2 print:border-black"
              >
                {/* Tree Number Badge */}
                <div className="bg-primary text-primary-foreground px-4 py-1 rounded-full text-sm font-bold mb-4">
                  Cây số {tree.tree_number}
                </div>

                {/* QR Code */}
                <div className="bg-white p-3 rounded-lg mb-4">
                  <QRCodeSVG
                    id={`qr-${tree.tree_number}`}
                    value={`${baseUrl}/tree/${tree.tree_number}`}
                    size={150}
                    level="H"
                    includeMargin={false}
                  />
                </div>

                {/* Tree Info */}
                <div className="text-center mb-3">
                  <p className="font-semibold text-foreground">{tree.species || "Chưa xác định"}</p>
                  {tree.location_description && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {tree.location_description}
                    </p>
                  )}
                </div>

                {/* URL Preview */}
                <p className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded font-mono print:text-[10px]">
                  /tree/{tree.tree_number}
                </p>

                {/* Download button - Hidden when printing */}
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-3 print:hidden"
                  onClick={() => downloadSingleQR(tree.tree_number)}
                >
                  <Download className="w-4 h-4 mr-1" />
                  Tải PNG
                </Button>
              </motion.div>
            ))}
          </div>
        )}

        {/* Print Footer */}
        <div className="hidden print:block mt-8 pt-4 border-t border-black text-center text-sm">
          <p className="font-bold">Tree-Math - Đo lường sức mạnh cây xanh</p>
          <p>Quét mã QR để xem bảng xếp hạng và tham gia đo cây!</p>
        </div>
      </div>

      {/* Print Styles */}
      <style>{`
        @media print {
          @page {
            size: A4;
            margin: 1cm;
          }
          body {
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
          }
        }
      `}</style>
    </div>
  );
};

export default QRCodesScreen;
