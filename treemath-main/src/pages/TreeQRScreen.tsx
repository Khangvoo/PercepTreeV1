import { useState, useEffect, useRef, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { TreeDeciduous, ArrowRight, Send, Bot, User, MessageCircle } from "lucide-react";
import FallingLeaves from "@/components/FallingLeaves";

interface LeaderboardEntry {
  id: string;
  user_name: string;
  user_class: string;
  accuracy_score: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

const INITIAL_BOT_MESSAGE = `Xin chào! 🌳 Bạn vừa quét mã QR của cây này rồi đấy!

Trước khi đo, bạn hãy **quay lại gian hàng Tree-Math** để lấy các dụng cụ đo nhé:
- 📏 Thước dây để đo chu vi
- 📐 Giác kế (clinometer) hoặc app điện thoại để đo chiều cao

Khi đã có dụng cụ, hãy bấm **"Tôi đã sẵn sàng"** hoặc hỏi tôi nếu cần hướng dẫn cách đo! 💪`;

const TYPEWRITER_TEXT = "Bạn có biết về sức mạnh của cây xanh? Mỗi cây có thể hấp thụ hàng chục kg CO₂ mỗi năm, góp phần làm mát không khí và bảo vệ môi trường. Hãy cùng đo và khám phá sức mạnh của cây này nhé! 🌍💚";

type ScreenPhase = "intro" | "chatbot";

const TreeQRScreen = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const treeNumber = useMemo(() => {
    const num = parseInt(id || "1");
    return isNaN(num) ? 1 : num;
  }, [id]);

  // Phase state
  const [phase, setPhase] = useState<ScreenPhase>("intro");
  
  // Intro phase states
  const [displayedText, setDisplayedText] = useState("");
  const [showLeaderboard, setShowLeaderboard] = useState(false);
  const [showMeasureButton, setShowMeasureButton] = useState(false);
  
  // Leaderboard data
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [leaderboardLoading, setLeaderboardLoading] = useState(true);

  // Chatbot phase states
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: INITIAL_BOT_MESSAGE }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Typewriter effect for intro - optimized with RAF
  useEffect(() => {
    if (phase !== "intro") return;
    
    let index = 0;
    let animationId: number;
    let lastTime = 0;
    const charDelay = 30; // ms per character
    
    const animate = (time: number) => {
      if (time - lastTime >= charDelay) {
        if (index < TYPEWRITER_TEXT.length) {
          setDisplayedText(TYPEWRITER_TEXT.slice(0, index + 1));
          index++;
          lastTime = time;
        } else {
          setTimeout(() => setShowLeaderboard(true), 300);
          return;
        }
      }
      animationId = requestAnimationFrame(animate);
    };
    
    animationId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationId);
  }, [phase]);

  // Show measure button after leaderboard appears
  useEffect(() => {
    if (showLeaderboard) {
      const timer = setTimeout(() => setShowMeasureButton(true), 500);
      return () => clearTimeout(timer);
    }
  }, [showLeaderboard]);

  // Fetch leaderboard data
  useEffect(() => {
    const fetchLeaderboard = async () => {
      if (isNaN(treeNumber)) return;
      
      setLeaderboardLoading(true);
      // Use masked RPC function to protect student privacy
      const { data, error } = await supabase
        .rpc('get_leaderboard_masked', { 
          p_tree_number: treeNumber, 
          p_limit: 5 
        });

      if (!error && data) {
        setLeaderboard(data as LeaderboardEntry[]);
      }
      setLeaderboardLoading(false);
    };

    fetchLeaderboard();
  }, [treeNumber]);

  // Scroll chat to bottom - debounced
  useEffect(() => {
    if (chatContainerRef.current) {
      requestAnimationFrame(() => {
        chatContainerRef.current?.scrollTo({
          top: chatContainerRef.current.scrollHeight,
          behavior: 'smooth'
        });
      });
    }
  }, [messages]);

  const handleStartMeasure = () => {
    setPhase("chatbot");
  };

  const sendMessage = async (message: string) => {
    if (!message.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: "user", content: message };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputMessage("");
    setIsLoading(true);

    try {
      const CHAT_URL = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/tree-measurement-chat`;
      
      const resp = await fetch(CHAT_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
        },
        body: JSON.stringify({ messages: updatedMessages }),
      });

      if (!resp.ok || !resp.body) {
        throw new Error("Failed to get response");
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let textBuffer = "";
      let assistantContent = "";

      // Add empty assistant message first
      setMessages(prev => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        textBuffer += decoder.decode(value, { stream: true });

        let newlineIndex: number;
        while ((newlineIndex = textBuffer.indexOf("\n")) !== -1) {
          let line = textBuffer.slice(0, newlineIndex);
          textBuffer = textBuffer.slice(newlineIndex + 1);

          if (line.endsWith("\r")) line = line.slice(0, -1);
          if (line.startsWith(":") || line.trim() === "") continue;
          if (!line.startsWith("data: ")) continue;

          const jsonStr = line.slice(6).trim();
          if (jsonStr === "[DONE]") break;

          try {
            const parsed = JSON.parse(jsonStr);
            const content = parsed.choices?.[0]?.delta?.content as string | undefined;
            if (content) {
              assistantContent += content;
              setMessages(prev => {
                const newMessages = [...prev];
                newMessages[newMessages.length - 1] = { role: "assistant", content: assistantContent };
                return newMessages;
              });
            }
          } catch {
            textBuffer = line + "\n" + textBuffer;
            break;
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại nhé! 🙏"
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoToMeasure = () => {
    navigate(`/map?tree=${treeNumber}`);
  };

  const formatMessage = (content: string | undefined) => {
    if (!content) return null;
    const lines = content.split("\n");
    return lines.map((line, i) => (
      <span key={i}>
        {line.split(/(\*\*.*?\*\*)/).map((part, j) => {
          if (part.startsWith("**") && part.endsWith("**")) {
            return <strong key={j}>{part.slice(2, -2)}</strong>;
          }
          return part;
        })}
        {i < lines.length - 1 && <br />}
      </span>
    ));
  };

  // Helper to get initials from name
  const getInitials = (name: string) => {
    return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
  };

  // INTRO PHASE
  if (phase === "intro") {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background via-forest/10 to-background flex flex-col relative overflow-hidden">
        <FallingLeaves />
        
        {/* Decorative background elements */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute top-20 left-10 w-40 h-40 bg-primary/15 rounded-full blur-3xl" />
          <div className="absolute bottom-32 right-5 w-48 h-48 bg-leaf/15 rounded-full blur-3xl" />
          <div className="absolute top-1/3 right-1/4 w-32 h-32 bg-gold/10 rounded-full blur-3xl" />
        </div>

        {/* Header with Tree Illustration */}
        <div className="flex flex-col items-center justify-center pt-8 pb-4 px-4 animate-fade-in relative z-10">
          <div className="relative mb-4">
            {/* Outer glow ring */}
            <div className="absolute inset-[-8px] bg-gradient-to-br from-primary/40 to-forest/30 rounded-full blur-xl animate-pulse" />
            {/* Main icon container */}
            <div className="relative w-24 h-24 rounded-full bg-gradient-to-br from-primary/40 via-forest/30 to-leaf/30 flex items-center justify-center border-2 border-primary/50 shadow-2xl">
              <TreeDeciduous className="w-12 h-12 text-primary drop-shadow-lg" />
            </div>
            {/* Sparkle accents */}
            <div className="absolute -top-1 -right-1 w-4 h-4 bg-gold rounded-full blur-sm animate-pulse" />
            <div className="absolute -bottom-1 -left-1 w-3 h-3 bg-leaf rounded-full blur-sm animate-pulse" style={{ animationDelay: '0.5s' }} />
          </div>
          
          {/* Tree number badge */}
          <div className="inline-flex items-center gap-3 bg-gradient-to-r from-primary/30 via-forest/25 to-primary/30 text-primary px-6 py-3 rounded-full shadow-lg border border-primary/40">
            <span className="text-2xl">🌳</span>
            <span className="text-xl font-bold tracking-wide">Cây số {treeNumber}</span>
          </div>
        </div>

        <div className="flex-1 px-5 flex flex-col relative z-10 overflow-y-auto">
          {/* CO2 Highlight Card */}
          <div className="glass-premium p-6 mb-6 animate-fade-in">
            {/* Main stat highlight */}
            <div className="flex items-center justify-center gap-4 mb-4">
              <div className="stat-highlight">
                <span className="text-4xl font-extrabold text-primary text-glow">20-50</span>
                <div className="flex flex-col">
                  <span className="text-lg font-bold text-foreground">kg CO₂</span>
                  <span className="text-sm text-muted-foreground">mỗi năm</span>
                </div>
              </div>
            </div>
            
            {/* Description text */}
            <p className="text-foreground text-lg leading-relaxed text-center">
              {displayedText}
              {displayedText.length < TYPEWRITER_TEXT.length && (
                <span className="animate-pulse text-primary ml-0.5">|</span>
              )}
            </p>
          </div>

          {/* Leaderboard */}
          {showLeaderboard && (
            <div className="glass-premium p-5 mb-6 animate-fade-in">
              <h3 className="text-foreground font-bold text-center mb-5 text-xl flex items-center justify-center gap-3">
                <span className="text-3xl">🏆</span>
                <span>Bảng xếp hạng</span>
              </h3>

              {leaderboardLoading ? (
                <div className="text-muted-foreground text-center py-6 text-base">Đang tải...</div>
              ) : leaderboard.length === 0 ? (
                <div className="text-center py-6">
                  <div className="text-4xl mb-3">🎯</div>
                  <p className="text-muted-foreground text-base">
                    Chưa có ai đo cây này
                  </p>
                  <p className="text-primary font-semibold text-lg mt-1">
                    Hãy là người đầu tiên!
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {leaderboard.slice(0, 3).map((entry, index) => (
                    <div
                      key={entry.id}
                      className={`flex items-center gap-4 p-4 rounded-2xl transition-all duration-300 hover:scale-[1.02] ${
                        index === 0 ? "rank-gold scale-[1.02]" :
                        index === 1 ? "rank-silver" :
                        "rank-bronze"
                      }`}
                    >
                      {/* Medal Badge */}
                      <div className={`relative flex-shrink-0 ${index === 0 ? 'scale-110' : ''}`}>
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center text-3xl ${
                          index === 0 ? 'glow-gold' : ''
                        }`}>
                          {index === 0 ? "🥇" : index === 1 ? "🥈" : "🥉"}
                        </div>
                      </div>
                      
                      {/* Avatar */}
                      <div className={`avatar-ring w-11 h-11 flex-shrink-0 ${
                        index === 0 ? 'border-gold/70 bg-gradient-to-br from-gold/30 to-gold/10' : 
                        index === 1 ? 'border-silver/70 bg-gradient-to-br from-silver/30 to-silver/10' :
                        'border-bronze/70 bg-gradient-to-br from-bronze/30 to-bronze/10'
                      }`}>
                        <span className={`${
                          index === 0 ? 'text-gold' : 
                          index === 1 ? 'text-silver' : 
                          'text-bronze'
                        } font-bold`}>
                          {getInitials(entry.user_name)}
                        </span>
                      </div>
                      
                      {/* Info */}
                      <div className="flex-1 min-w-0">
                        <p className={`font-bold truncate ${
                          index === 0 ? 'text-lg text-glow-gold text-gold' : 'text-base text-foreground'
                        }`}>
                          {entry.user_name}
                        </p>
                        <p className="text-muted-foreground text-sm truncate">{entry.user_class}</p>
                      </div>
                      
                      {/* Score */}
                      <div className={`px-4 py-2 rounded-xl ${
                        index === 0 
                          ? 'bg-gradient-to-r from-gold/30 to-gold/20 border border-gold/40' 
                          : 'bg-primary/20 border border-primary/30'
                      }`}>
                        <span className={`font-extrabold text-lg ${
                          index === 0 ? 'text-gold text-glow-gold' : 'text-primary'
                        }`}>
                          {entry.accuracy_score.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* CTA Button */}
        {showMeasureButton && (
          <div className="p-6 animate-fade-in relative z-10">
            <button
              onClick={handleStartMeasure}
              className="btn-cta w-full py-5 text-xl text-primary-foreground flex items-center justify-center gap-3 animate-breathe"
            >
              <MessageCircle className="w-6 h-6" />
              <span>Tôi muốn đo cây này</span>
              <ArrowRight className="w-6 h-6" />
            </button>
          </div>
        )}
      </div>
    );
  }

  // CHATBOT PHASE
  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-forest/10 to-background flex flex-col relative overflow-hidden">
      <FallingLeaves />
      
      {/* Decorative background elements */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-20 right-10 w-24 h-24 bg-primary/10 rounded-full blur-2xl" />
        <div className="absolute bottom-40 left-10 w-32 h-32 bg-leaf/10 rounded-full blur-2xl" />
      </div>

      {/* Header */}
      <div className="flex items-center justify-center gap-4 pt-6 pb-5 px-4 animate-fade-in relative z-10 bg-gradient-to-b from-background/80 to-transparent backdrop-blur-sm">
        <div className="relative">
          <div className="absolute inset-0 bg-primary/20 rounded-full blur-lg" />
          <div className="relative w-14 h-14 rounded-full bg-gradient-to-br from-primary/30 to-forest/30 flex items-center justify-center border-2 border-primary/40 shadow-lg">
            <Bot className="w-7 h-7 text-primary" />
          </div>
        </div>
        <div>
          <h1 className="text-foreground font-bold text-xl">Trợ lý đo cây</h1>
          <span className="text-muted-foreground text-base">🌳 Cây số {treeNumber}</span>
        </div>
      </div>

      {/* Chat Container */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto px-4 pb-4 space-y-5 relative z-10"
      >
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex gap-3 animate-fade-in ${msg.role === "user" ? "flex-row-reverse" : ""}`}
          >
            <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-md ${
              msg.role === "assistant" 
                ? "bg-gradient-to-br from-primary/30 to-forest/30 border border-primary/30" 
                : "bg-gradient-to-br from-muted to-muted/80 border border-border/50"
            }`}>
              {msg.role === "assistant" ? (
                <Bot className="w-5 h-5 text-primary" />
              ) : (
                <User className="w-5 h-5 text-foreground" />
              )}
            </div>
            <div className={`max-w-[80%] rounded-3xl px-5 py-4 shadow-lg ${
              msg.role === "assistant" 
                ? "bg-gradient-to-br from-card/90 to-card/70 text-foreground border border-border/50 rounded-tl-lg" 
                : "bg-gradient-to-r from-primary to-forest text-primary-foreground rounded-tr-lg"
            }`}>
              <p className="text-base leading-relaxed">
                {formatMessage(msg.content)}
              </p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-3 animate-fade-in">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary/30 to-forest/30 flex items-center justify-center border border-primary/30 shadow-md">
              <Bot className="w-5 h-5 text-primary" />
            </div>
            <div className="bg-gradient-to-br from-card/90 to-card/70 rounded-3xl rounded-tl-lg px-5 py-4 shadow-lg border border-border/50">
              <div className="flex gap-1.5">
                <span className="w-2.5 h-2.5 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-2.5 h-2.5 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-2.5 h-2.5 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 space-y-4 animate-fade-in relative z-10 bg-gradient-to-t from-background via-background/95 to-transparent pt-6">
        {/* Quick Actions */}
        <div className="flex gap-3 overflow-x-auto pb-2">
          <Button
            size="default"
            variant="outline"
            onClick={handleGoToMeasure}
            className="whitespace-nowrap bg-gradient-to-r from-primary/25 to-forest/20 border-primary/50 text-foreground hover:from-primary/35 hover:to-forest/30 rounded-2xl px-5 py-3 text-base font-semibold shadow-md transition-all duration-300 hover:scale-[1.02]"
          >
            ✅ Tôi đã sẵn sàng
          </Button>
          <Button
            size="default"
            variant="outline"
            onClick={() => sendMessage("Hướng dẫn đo chiều cao cây")}
            className="whitespace-nowrap bg-muted/80 border-border text-foreground hover:bg-muted rounded-2xl px-5 py-3 text-base shadow-md transition-all duration-300 hover:scale-[1.02]"
          >
            📏 Đo chiều cao
          </Button>
          <Button
            size="default"
            variant="outline"
            onClick={() => sendMessage("Hướng dẫn đo chu vi cây")}
            className="whitespace-nowrap bg-muted/80 border-border text-foreground hover:bg-muted rounded-2xl px-5 py-3 text-base shadow-md transition-all duration-300 hover:scale-[1.02]"
          >
            🔄 Đo chu vi
          </Button>
        </div>

        {/* Chat Input */}
        <div className="flex gap-3">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && sendMessage(inputMessage)}
            placeholder="Hỏi tôi về cách đo cây..."
            className="flex-1 bg-input/80 border-2 border-border/50 rounded-2xl px-5 py-4 text-base text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary focus:bg-input transition-all duration-300 shadow-inner"
          />
          <Button
            onClick={() => sendMessage(inputMessage)}
            disabled={!inputMessage.trim() || isLoading}
            className="bg-gradient-to-r from-primary to-forest hover:from-primary/90 hover:to-forest/90 rounded-2xl px-5 shadow-lg transition-all duration-300 hover:scale-105 disabled:opacity-50"
          >
            <Send className="w-6 h-6" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default TreeQRScreen;
