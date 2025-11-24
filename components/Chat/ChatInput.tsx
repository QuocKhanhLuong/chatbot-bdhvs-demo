import { Message } from "@/types";
import { FC, KeyboardEvent, useEffect, useRef, useState } from "react";
import { Send, Loader2 } from "lucide-react";

interface Props {
  onSend: (message: Message) => void;
  disabled?: boolean;
  darkMode?: boolean;
}

export const ChatInput: FC<Props> = ({ onSend, disabled = false, darkMode = false }) => {
  const [content, setContent] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [lineCount, setLineCount] = useState(1);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setError(null);

    if (value.length > 4000) {
      setError("Message cannot exceed 4000 characters");
      return;
    }

    setContent(value);
  };

  const handleSend = () => {
    if (!content.trim() || disabled) {
      return;
    }

    // Check one more time to ensure the message is not too long
    if (content.length > 4000) {
      setError("Message cannot exceed 4000 characters");
      return;
    }

    onSend({ role: "user", content });
    setContent("");
    setError(null);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      const textarea = textareaRef.current;
      const lineHeight = 20; // Approximate line height in pixels
      const minHeight = 44;

      // Reset height to calculate scroll height
      textarea.style.height = "auto";
      const scrollHeight = textarea.scrollHeight;
      const calculatedLines = Math.max(1, Math.floor((scrollHeight - 16) / lineHeight));

      setLineCount(calculatedLines);

      if (calculatedLines <= 8) {
        // Auto-expand for 1-8 lines
        textarea.style.height = `${Math.max(minHeight, scrollHeight)}px`;
        textarea.style.overflowY = "hidden";
      } else {
        // Fixed height with scroll for 8+ lines
        textarea.style.height = `${minHeight + 7 * lineHeight}px`;
        textarea.style.overflowY = "auto";
      }
    }
  }, [content]);

  const hasContent = content.length > 0;

  return (
    <div className="flex flex-col">
      <div
        className={`flex flex-col rounded-2xl border bg-white shadow-sm transition-all duration-200 dark:bg-gray-900 ${
          error
            ? "border-red-500 ring-1 ring-red-500"
            : hasContent
            ? "border-pink-300 ring-1 ring-pink-300 dark:border-pink-700"
            : "border-gray-300 dark:border-gray-600"
        } p-3`}
      >
        <div className="relative flex-1">
          <textarea
            ref={textareaRef}
            value={content}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="How can I help you today?"
            rows={1}
            disabled={disabled}
            className={`w-full resize-none bg-transparent px-0 py-2 text-sm text-gray-900 outline-none placeholder:text-gray-400 transition-all duration-200 dark:text-white dark:placeholder:text-gray-500 ${
              disabled ? "cursor-not-allowed opacity-50" : ""
            }`}
            style={{
              minHeight: "44px",
              height: "auto",
              overflowY: lineCount > 8 ? "auto" : "hidden",
            }}
          />
        </div>

        <div className="mt-2 flex items-center justify-end">
          <button
            onClick={handleSend}
            disabled={disabled || !content.trim() || !!error}
            className={`inline-flex shrink-0 items-center gap-2 rounded-full bg-gradient-to-r from-pink-500 to-rose-500 px-4 py-2 text-sm font-medium text-white shadow-sm transition-all hover:from-pink-600 hover:to-rose-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-pink-500 focus-visible:ring-offset-2 ${
              disabled || !content.trim() || !!error ? "cursor-not-allowed opacity-50" : "hover:scale-105"
            }`}
          >
            {disabled ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            <span>Send</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-2 px-2 text-xs text-red-500">{error}</div>
      )}

      <div className="mt-2 px-2 text-xs text-gray-500 dark:text-gray-400">
        Press{" "}
        <kbd className="rounded border border-gray-300 bg-gray-50 px-1 text-[10px] dark:border-gray-600 dark:bg-gray-800">
          Enter
        </kbd>{" "}
        to send ·{" "}
        <kbd className="rounded border border-gray-300 bg-gray-50 px-1 text-[10px] dark:border-gray-600 dark:bg-gray-800">
          Shift
        </kbd>
        +
        <kbd className="rounded border border-gray-300 bg-gray-50 px-1 text-[10px] dark:border-gray-600 dark:bg-gray-800">
          Enter
        </kbd>{" "}
        for newline
      </div>
    </div>
  );
};
