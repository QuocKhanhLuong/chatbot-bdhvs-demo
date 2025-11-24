import { Message } from "@/types";
import { FC, useEffect, useState, useMemo } from "react";
import { CodeBlock } from "./CodeBlock";
import ReactMarkdown from "react-markdown";

interface Props {
  message: Message;
  isStreaming?: boolean;
  darkMode?: boolean;
}

export const ChatMessage: FC<Props> = ({ message, isStreaming = false, darkMode = false }) => {
  const [displayedContent, setDisplayedContent] = useState<string>(message.content || "");
  const [isNew, setIsNew] = useState<boolean>(true);

  // Update displayed content when the message content changes (during streaming)
  useEffect(() => {
    setDisplayedContent(message.content || "");
  }, [message.content]);

  // Reset the "new" state after animation completes
  useEffect(() => {
    if (isNew) {
      const timer = setTimeout(() => setIsNew(false), 500);
      return () => clearTimeout(timer);
    }
  }, [isNew]);

  // Format the message content with proper handling of code blocks
  const formattedContent = useMemo(() => {
    if (!displayedContent) return [<div key="empty"></div>];

    const parseCodeBlocks = (text: string) => {
      const segments = text.split(/(```[\s\S]*?```)/);
      return segments.map((segment, index) => {
        if (segment.startsWith("```") && segment.endsWith("```")) {
          // Extract code content
          const codeContent = segment.slice(3, -3);
          const firstLineBreak = codeContent.indexOf("\n");

          if (firstLineBreak === -1) {
            return (
              <div key={`code-${index}`} className="mb-4 last:mb-0">
                <CodeBlock language="plaintext" value={codeContent.trim()} />
              </div>
            );
          }

          const firstLine = codeContent.slice(0, firstLineBreak).trim();
          const restOfCode = codeContent.slice(firstLineBreak + 1);

          return (
            <div key={`code-${index}`} className="mb-4 last:mb-0">
              <CodeBlock language={firstLine || "plaintext"} value={restOfCode.trim()} />
            </div>
          );
        } else if (segment.trim()) {
          // Regular text
          return (
            <ReactMarkdown key={`text-${index}`}>
              {segment}
            </ReactMarkdown>
          );
        }

        return null;
      }).filter(Boolean);
    };

    return parseCodeBlocks(displayedContent);
  }, [displayedContent]);

  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"} ${isNew ? 'animate-fadeIn' : ''}`}>
      {/* Avatar for Assistant */}
      {!isUser && (
        <div className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-gradient-to-br from-pink-500 to-rose-500 text-xs font-bold text-white shadow-sm">
          AI
        </div>
      )}

      {/* Message bubble */}
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm shadow-sm transition-all ${
          isUser
            ? "bg-gradient-to-br from-pink-500 to-rose-500 text-white"
            : darkMode
            ? "border border-gray-700 bg-gray-800 text-gray-100"
            : "border border-pink-100 bg-white text-gray-900"
        }`}
        style={{ overflowWrap: "anywhere" }}
      >
        <div className="whitespace-pre-wrap break-words">
          {formattedContent}
        </div>
        {isStreaming && (
          <span className="ml-1 inline-block h-3 w-1 animate-pulse bg-current"></span>
        )}
      </div>

      {/* Avatar for User */}
      {isUser && (
        <div className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-gradient-to-br from-blue-500 to-indigo-500 text-xs font-bold text-white shadow-sm">
          U
        </div>
      )}
    </div>
  );
};
