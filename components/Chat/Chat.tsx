import { Message } from "@/types";
import { FC, useEffect, useRef, useState } from "react";
import { ChatInput } from "./ChatInput";
import { ChatLoader } from "./ChatLoader";
import { ChatMessage } from "./ChatMessage";
import { ResetChat } from "./ResetChat";
import { useDarkMode } from "../Contexts/DarkModeContext";

interface Props {
  messages: Message[];
  loading: boolean;
  streaming: boolean;
  onSend: (message: Message) => void;
  onReset: () => void;
}

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
          <div className="text-2xl sm:text-3xl font-sans tracking-tight text-gray-900 dark:text-white">
            Chatbot Soni
          </div>
          <ResetChat onReset={onReset} />
        </div>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {count} {count === 1 ? 'message' : 'messages'}
        </div>
        
        {/* Tags */}
        <div className="mt-4 flex flex-wrap gap-2 border-b border-gray-200 pb-4 dark:border-gray-700">
          <span className="inline-flex items-center rounded-full border border-pink-200 bg-pink-50 px-3 py-1 text-xs text-pink-700 dark:border-pink-800 dark:bg-pink-900/30 dark:text-pink-300">
            Certified
          </span>
          <span className="inline-flex items-center rounded-full border border-pink-200 bg-pink-50 px-3 py-1 text-xs text-pink-700 dark:border-pink-800 dark:bg-pink-900/30 dark:text-pink-300">
            Personalized
          </span>
          <span className="inline-flex items-center rounded-full border border-pink-200 bg-pink-50 px-3 py-1 text-xs text-pink-700 dark:border-pink-800 dark:bg-pink-900/30 dark:text-pink-300">
            Experienced
          </span>
          <span className="inline-flex items-center rounded-full border border-pink-200 bg-pink-50 px-3 py-1 text-xs text-pink-700 dark:border-pink-800 dark:bg-pink-900/30 dark:text-pink-300">
            Helpful
          </span>
        </div>
      </div>

      {/* Messages area */}
      <div
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-8"
      >
        {messages.length === 0 ? (
          <div className="rounded-xl border border-dashed border-pink-300 bg-white p-6 text-sm text-gray-500 dark:border-pink-700 dark:bg-gray-800 dark:text-gray-400">
            No messages yet. Say hello to start.
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
