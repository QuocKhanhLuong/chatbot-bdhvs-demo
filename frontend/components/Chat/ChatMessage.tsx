import { Message } from "@/types";
import { FC, useEffect, useState, useMemo } from "react";
import { CodeBlock } from "./CodeBlock";
import ReactMarkdown from "react-markdown";
import { ChevronDown, ChevronRight, Loader2, Search, Brain, FileText, Check, ExternalLink, Code } from "lucide-react";
import type { Artifact } from "../Artifacts/ArtifactPanel";

interface StatusUpdate {
  type: "status" | "error";
  stage: string;
  message: string;
  progress?: number;
  current_query?: string;
  depth?: number;
  breadth?: number;
}

interface Props {
  message: Message;
  isStreaming?: boolean;
  darkMode?: boolean;
  onViewArtifact?: (artifact: Artifact) => void;
}

// Parse status updates from message content
function parseStatusUpdates(content: string): { statusUpdates: StatusUpdate[]; cleanContent: string } {
  const statusUpdates: StatusUpdate[] = [];
  const lines = content.split('\n');
  const cleanLines: string[] = [];
  
  for (const line of lines) {
    // Match [STATUS] prefix pattern
    const statusMatch = line.match(/^\[STATUS\]\s*(.+)$/);
    if (statusMatch) {
      try {
        // Try to parse as JSON first
        const jsonMatch = statusMatch[1].match(/^\{.*\}$/);
        if (jsonMatch) {
          statusUpdates.push(JSON.parse(jsonMatch[0]));
        } else {
          // Parse as simple status message
          statusUpdates.push({
            type: "status",
            stage: "processing",
            message: statusMatch[1]
          });
        }
      } catch {
        statusUpdates.push({
          type: "status",
          stage: "processing",
          message: statusMatch[1]
        });
      }
    } else {
      cleanLines.push(line);
    }
  }
  
  return {
    statusUpdates,
    cleanContent: cleanLines.join('\n').trim()
  };
}

// Artifact markers for detecting reports and code artifacts
const ARTIFACT_MARKERS = {
  reportStart: /---REPORT START---/,
  reportEnd: /---REPORT END---/,
  codeStart: /---CODE START---/,
  codeEnd: /---CODE END---/,
};

// Parse artifacts from content
function parseArtifacts(content: string): { 
  artifacts: Artifact[]; 
  cleanContent: string;
  hasLongReport: boolean;
} {
  const artifacts: Artifact[] = [];
  let cleanContent = content;
  let hasLongReport = false;

  // Check for report artifacts
  const reportMatch = content.match(/---REPORT START---\s*([\s\S]*?)\s*---REPORT END---/);
  if (reportMatch) {
    const reportContent = reportMatch[1].trim();
    // Consider it a "long report" if content is > 1000 chars
    hasLongReport = reportContent.length > 1000;
    
    artifacts.push({
      id: `report-${Date.now()}`,
      type: "report",
      title: extractReportTitle(reportContent) || "Research Report",
      content: reportContent,
      timestamp: Date.now(),
    });
    
    // Replace artifact marker in content with placeholder
    cleanContent = cleanContent.replace(
      /---REPORT START---\s*[\s\S]*?\s*---REPORT END---/,
      hasLongReport ? "[📄 Report generated - click to view in panel]" : ""
    );
  }

  // Check for code artifacts
  const codeMatch = content.match(/---CODE START---\s*([\s\S]*?)\s*---CODE END---/);
  if (codeMatch) {
    const codeContent = codeMatch[1].trim();
    
    artifacts.push({
      id: `code-${Date.now()}`,
      type: "code",
      title: extractCodeTitle(codeContent) || "Generated Code",
      content: codeContent,
      timestamp: Date.now(),
    });
    
    // For code, we still show it inline but also make it viewable in panel
    cleanContent = cleanContent.replace(
      /---CODE START---\s*[\s\S]*?\s*---CODE END---/,
      codeContent.length > 500 ? "[💻 Code generated - click to view in panel]" : codeContent
    );
  }

  return { artifacts, cleanContent: cleanContent.trim(), hasLongReport };
}

// Extract title from report content (first heading or first line)
function extractReportTitle(content: string): string | null {
  // Look for first markdown heading
  const headingMatch = content.match(/^#\s+(.+)$/m);
  if (headingMatch) return headingMatch[1].trim();
  
  // Look for bold title
  const boldMatch = content.match(/^\*\*(.+?)\*\*/);
  if (boldMatch) return boldMatch[1].trim();
  
  // Use first line if short enough
  const firstLine = content.split('\n')[0].trim();
  if (firstLine && firstLine.length <= 100) return firstLine;
  
  return null;
}

// Extract title from code content (function name or first comment)
function extractCodeTitle(content: string): string | null {
  // Look for function definition
  const funcMatch = content.match(/def\s+(\w+)/);
  if (funcMatch) return `Function: ${funcMatch[1]}`;
  
  // Look for class definition
  const classMatch = content.match(/class\s+(\w+)/);
  if (classMatch) return `Class: ${classMatch[1]}`;
  
  // Look for first comment
  const commentMatch = content.match(/^#\s*(.+)$/m);
  if (commentMatch) return commentMatch[1].trim();
  
  return null;
}

// Get icon for status stage
function getStageIcon(stage: string, isActive: boolean) {
  const baseClass = isActive ? "animate-spin" : "";
  
  switch (stage) {
    case "searching":
    case "searching_web":
    case "searching_arxiv":
      return <Search className={`h-3 w-3 ${isActive ? 'animate-pulse' : ''}`} />;
    case "processing":
    case "generating_queries":
    case "reflecting":
      return <Brain className={`h-3 w-3 ${isActive ? 'animate-pulse' : ''}`} />;
    case "generating_report":
    case "completed":
      return isActive ? <Loader2 className={`h-3 w-3 ${baseClass}`} /> : <Check className="h-3 w-3 text-green-500" />;
    default:
      return isActive ? <Loader2 className={`h-3 w-3 ${baseClass}`} /> : <FileText className="h-3 w-3" />;
  }
}

// Thinking Process Accordion Component
const ThinkingProcess: FC<{ statusUpdates: StatusUpdate[]; isStreaming: boolean; darkMode: boolean }> = ({ 
  statusUpdates, 
  isStreaming,
  darkMode 
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  
  if (statusUpdates.length === 0) return null;
  
  return (
    <div className={`mb-3 rounded-lg border ${darkMode ? 'border-gray-600 bg-gray-700/50' : 'border-pink-200 bg-pink-50/50'}`}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`flex w-full items-center gap-2 px-3 py-2 text-xs font-medium transition-colors ${
          darkMode ? 'text-gray-300 hover:bg-gray-600/50' : 'text-gray-600 hover:bg-pink-100/50'
        } rounded-t-lg`}
      >
        {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <Brain className="h-3 w-3" />
        <span>Thinking Process</span>
        {isStreaming && <Loader2 className="h-3 w-3 animate-spin ml-auto" />}
        {!isStreaming && <Check className="h-3 w-3 text-green-500 ml-auto" />}
      </button>
      
      {/* Content */}
      {isExpanded && (
        <div className={`px-3 pb-2 space-y-1.5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {statusUpdates.map((update, index) => {
            const isLast = index === statusUpdates.length - 1;
            const isActive = isLast && isStreaming;
            
            return (
              <div 
                key={index} 
                className={`flex items-start gap-2 text-xs ${
                  isActive ? (darkMode ? 'text-gray-200' : 'text-gray-700') : ''
                }`}
              >
                <span className="mt-0.5 shrink-0">
                  {getStageIcon(update.stage, isActive)}
                </span>
                <span className="leading-relaxed">
                  {update.message}
                  {update.current_query && (
                    <span className={`ml-1 italic ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      "{update.current_query.slice(0, 40)}..."
                    </span>
                  )}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export const ChatMessage: FC<Props> = ({ message, isStreaming = false, darkMode = false, onViewArtifact }) => {
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

  // Parse status updates and clean content
  const { statusUpdates, cleanContent } = useMemo(() => {
    return parseStatusUpdates(displayedContent);
  }, [displayedContent]);

  // Parse artifacts from content
  const { artifacts, cleanContent: contentWithoutArtifacts, hasLongReport } = useMemo(() => {
    return parseArtifacts(cleanContent);
  }, [cleanContent]);

  // Format the message content with proper handling of code blocks
  const formattedContent = useMemo(() => {
    if (!contentWithoutArtifacts) return [<div key="empty"></div>];

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

    return parseCodeBlocks(contentWithoutArtifacts);
  }, [contentWithoutArtifacts]);

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
        {/* Thinking Process Accordion - only show for assistant messages with status updates */}
        {!isUser && statusUpdates.length > 0 && (
          <ThinkingProcess 
            statusUpdates={statusUpdates} 
            isStreaming={isStreaming} 
            darkMode={darkMode}
          />
        )}
        
        {/* Main content */}
        <div className="whitespace-pre-wrap break-words">
          {formattedContent}
        </div>
        
        {/* Artifact Cards - clickable cards for reports/code */}
        {!isUser && artifacts.length > 0 && !isStreaming && (
          <div className="mt-3 space-y-2">
            {artifacts.map((artifact) => (
              <button
                key={artifact.id}
                onClick={() => onViewArtifact?.(artifact)}
                className={`flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-all hover:scale-[1.02] ${
                  darkMode
                    ? "border-gray-600 bg-gray-700/50 hover:bg-gray-600/50"
                    : "border-pink-200 bg-pink-50/50 hover:bg-pink-100/50"
                }`}
              >
                {/* Artifact Icon */}
                <div className={`shrink-0 rounded-md p-2 ${
                  artifact.type === "report"
                    ? darkMode ? "bg-emerald-900/50 text-emerald-400" : "bg-emerald-100 text-emerald-600"
                    : artifact.type === "code"
                    ? darkMode ? "bg-blue-900/50 text-blue-400" : "bg-blue-100 text-blue-600"
                    : darkMode ? "bg-pink-900/50 text-pink-400" : "bg-pink-100 text-pink-600"
                }`}>
                  {artifact.type === "report" ? (
                    <FileText className="h-5 w-5" />
                  ) : artifact.type === "code" ? (
                    <Code className="h-5 w-5" />
                  ) : (
                    <FileText className="h-5 w-5" />
                  )}
                </div>
                
                {/* Artifact Info */}
                <div className="min-w-0 flex-1">
                  <div className={`font-medium truncate ${darkMode ? "text-white" : "text-gray-900"}`}>
                    {artifact.title}
                  </div>
                  <div className={`text-xs ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
                    {artifact.type === "report" ? "Research Report" : "Generated Code"} • Click to view
                  </div>
                </div>
                
                {/* Open Icon */}
                <ExternalLink className={`h-4 w-4 shrink-0 ${darkMode ? "text-gray-500" : "text-gray-400"}`} />
              </button>
            ))}
          </div>
        )}
        
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
