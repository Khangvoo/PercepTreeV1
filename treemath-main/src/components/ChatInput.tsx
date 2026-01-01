import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Send, Ruler, TreePine, Calculator, HelpCircle } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
}

const quickReplies = [
  { label: 'Đo chiều cao', icon: TreePine, message: 'Làm sao để đo chiều cao cây?' },
  { label: 'Đo chu vi', icon: Ruler, message: 'Hướng dẫn đo chu vi thân cây' },
  { label: 'Tính đường kính', icon: Calculator, message: 'Công thức tính đường kính từ chu vi' },
  { label: 'Tips đo đạc', icon: HelpCircle, message: 'Cho tôi một số tips để đo cây chính xác hơn' },
];

const ChatInput = ({ onSend, isLoading }: ChatInputProps) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleQuickReply = (message: string) => {
    if (!isLoading) {
      onSend(message);
    }
  };

  return (
    <div className="space-y-3">
      {/* Quick replies */}
      <div className="flex flex-wrap gap-2">
        {quickReplies.map((reply) => (
          <Button
            key={reply.label}
            type="button"
            variant="outline"
            size="sm"
            onClick={() => handleQuickReply(reply.message)}
            disabled={isLoading}
            className="text-xs gap-1.5 border-border/50 hover:bg-primary/10 hover:border-primary/50"
          >
            <reply.icon className="w-3.5 h-3.5" />
            {reply.label}
          </Button>
        ))}
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Nhập câu hỏi về đo cây..."
          disabled={isLoading}
          className="flex-1 px-4 py-3 rounded-xl bg-input/50 border border-border/50 
                     focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/50
                     text-foreground placeholder:text-muted-foreground"
        />
        <Button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="px-4 bg-primary hover:bg-primary/90"
        >
          <Send className="w-5 h-5" />
        </Button>
      </form>
    </div>
  );
};

export default ChatInput;
