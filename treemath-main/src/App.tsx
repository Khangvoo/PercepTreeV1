import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import SplashScreen from "./pages/SplashScreen";
import MapScreen from "./pages/MapScreen";
import TreeQRScreen from "./pages/TreeQRScreen";
import QRCodesScreen from "./pages/QRCodesScreen";
import UserInfoScreen from "./pages/UserInfoScreen";
import MeasurementScreen from "./pages/MeasurementScreen";
import LoadingScreen from "./pages/LoadingScreen";
import CongratsScreen from "./pages/CongratsScreen";
import ResultsScreen from "./pages/ResultsScreen";
import LeaderboardScreen from "./pages/LeaderboardScreen";
import ChatScreen from "./pages/ChatScreen";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<SplashScreen />} />
          <Route path="/tree/:id" element={<TreeQRScreen />} />
          <Route path="/qr-codes" element={<QRCodesScreen />} />
          <Route path="/map" element={<MapScreen />} />
          <Route path="/user-info" element={<UserInfoScreen />} />
          <Route path="/measurement" element={<MeasurementScreen />} />
          <Route path="/loading" element={<LoadingScreen />} />
          <Route path="/congrats" element={<CongratsScreen />} />
          <Route path="/results" element={<ResultsScreen />} />
          <Route path="/leaderboard/:treeNumber?" element={<LeaderboardScreen />} />
          <Route path="/chat" element={<ChatScreen />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
