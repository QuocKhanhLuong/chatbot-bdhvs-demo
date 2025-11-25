import { Message } from "@/types";
import { FC, useEffect, useRef, useState } from "react";
import { ChatInput } from "./ChatInput";
import { ChatLoader } from "./ChatLoader";
import { ChatMessage } from "./ChatMessage";
import { ResetChat } from "./ResetChat";
import { useDarkMode } from "../Contexts/DarkModeContext";
import { Sparkles, Search, Code, FileText } from "lucide-react";

interface Props {
  messages: Message[];
  loading: boolean;
  streaming: boolean;
  onSend: (message: Message) => void;
  onReset: () => void;
}

// Suggestion chips data
const suggestionChips = [
  {
    label: "Analyze latest AI papers",
    prompt: "Nghiên cứu sâu về những paper AI mới nhất trong tháng này, đặc biệt về LLM và AI Agents",
    icon: FileText,
  },
  {
    label: "Research DeepSeek-R1",
    prompt: "Nghiên cứu chi tiết về DeepSeek-R1 model: kiến trúc, hiệu năng và so sánh với GPT-4o",
    icon: Search,
  },
  {
    label: "Python Data Analysis",
    prompt: "Hãy viết code Python để phân tích dữ liệu và tạo visualization với matplotlib",
    icon: Code,
  },
  {
    label: "Compare LLM Models",
    prompt: "So sánh các LLM models: GPT-4, Claude, Gemini, Llama 3 về hiệu năng và use cases",
    icon: Sparkles,
  },
];

export const Chat: FC<Props> = ({ messages, loading, streaming, onSend, onReset }) => {
  const { darkMode } = useDarkMode();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = (smooth: boolean = false) => {
    messagesEndRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto" });
  };

  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollButton(!isNearBottom);
    }
  };

  // Handle suggestion chip click
  const handleSuggestionClick = (prompt: string) => {
    if (loading || streaming) return;
    onSend({
      role: "user",
      content: prompt,
    });
  };

  useEffect(() => {
    // Always scroll to bottom when streaming or new messages arrive
    scrollToBottom(true);
  }, [messages, loading, streaming]);

  const count = messages.length;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col">
      {/* Header with title and reset button */}
      <div className="flex-none px-4 py-4 sm:px-8">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-pink-500" />
            <div className="text-2xl sm:text-3xl font-sans tracking-tight text-gray-900 dark:text-white">
              AI Research Assistant
            </div>
          </div>
          <ResetChat onReset={onReset} />
        </div>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {count} {count === 1 ? 'message' : 'messages'} • Deep Research • Web Search • Code Execution
        </div>
        
        {/* Suggestion Chips - Functional buttons */}
        <div className="mt-4 flex flex-wrap gap-2 border-b border-gray-200 pb-4 dark:border-gray-700">
          {suggestionChips.map((chip, index) => {
            const Icon = chip.icon;
            return (
              <button
                key={index}
                onClick={() => handleSuggestionClick(chip.prompt)}
                disabled={loading || streaming}
                className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-all
                  ${loading || streaming 
                    ? 'cursor-not-allowed opacity-50' 
                    : 'cursor-pointer hover:scale-105 hover:shadow-md'
                  }
                  ${darkMode 
                    ? 'border-pink-800 bg-pink-900/30 text-pink-300 hover:bg-pink-800/50' 
                    : 'border-pink-200 bg-pink-50 text-pink-700 hover:bg-pink-100'
                  }`}
              >
                <Icon className="h-3 w-3" />
                {chip.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Messages area */}
      <div
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-8"
      >
        {messages.length === 0 ? (
          <div className="rounded-xl border border-dashed border-pink-300 bg-white p-6 text-center dark:border-pink-700 dark:bg-gray-800">
            <Sparkles className="mx-auto h-10 w-10 text-pink-400 mb-3" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              Welcome to AI Research Assistant
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              I can help you with deep research, web search, code execution, and more.
              <br />
              Try clicking one of the suggestion chips above or type your question below.
            </p>
            <div className="flex flex-wrap justify-center gap-2 text-xs text-gray-400 dark:text-gray-500">
              <span className="flex items-center gap-1"><Search className="h-3 w-3" /> Web Search</span>
              <span>•</span>
              <span className="flex items-center gap-1"><FileText className="h-3 w-3" /> ArXiv Papers</span>
              <span>•</span>
              <span className="flex items-center gap-1"><Code className="h-3 w-3" /> Python Execution</span>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <div key={`msg-${index}-${message.timestamp || index}`}>
                <ChatMessage
                  message={message}
                  isStreaming={message.isStreaming && streaming}
                  darkMode={darkMode}
                />
              </div>
            ))}
          </>
        )}

        {loading && !streaming && (
          <div className="my-1 sm:my-1.5">
            <ChatLoader />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="flex-none border-t border-gray-200/60 p-4 dark:border-gray-700">
        <ChatInput onSend={onSend} disabled={loading || streaming} darkMode={darkMode} />
      </div>

      {/* Scroll to bottom button */}
      {showScrollButton && (
        <button
          onClick={() => scrollToBottom(true)}
          className="fixed bottom-24 right-6 rounded-full bg-pink-500 p-3 text-white shadow-lg transition-all duration-200 hover:scale-110 hover:bg-pink-600 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:ring-offset-2"
          aria-label="Scroll to bottom"
        >
          <svg
            className="h-6 w-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
        </button>
      )}
    </div>
  );
};
