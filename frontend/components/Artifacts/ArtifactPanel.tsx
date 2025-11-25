"use client";

import { FC, useEffect, useState, useMemo } from "react";
import { X, Copy, Check, FileText, Code, ChevronDown, ChevronUp, Download } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { CodeBlock } from "../Chat/CodeBlock";

export interface Artifact {
  id: string;
  type: "report" | "code" | "document";
  title: string;
  content: string;
  timestamp?: number;
}

interface Props {
  artifact: Artifact | null;
  onClose: () => void;
  darkMode?: boolean;
}

// Get icon for artifact type
function getArtifactIcon(type: Artifact["type"]) {
  switch (type) {
    case "report":
      return <FileText className="h-5 w-5" />;
    case "code":
      return <Code className="h-5 w-5" />;
    default:
      return <FileText className="h-5 w-5" />;
  }
}

// Get title color for artifact type
function getArtifactColor(type: Artifact["type"], darkMode: boolean) {
  switch (type) {
    case "report":
      return darkMode ? "text-emerald-400" : "text-emerald-600";
    case "code":
      return darkMode ? "text-blue-400" : "text-blue-600";
    default:
      return darkMode ? "text-pink-400" : "text-pink-600";
  }
}

export const ArtifactPanel: FC<Props> = ({ artifact, onClose, darkMode = false }) => {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);

  // Handle copy to clipboard
  const handleCopy = async () => {
    if (!artifact) return;
    
    try {
      await navigator.clipboard.writeText(artifact.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  // Handle download as file
  const handleDownload = () => {
    if (!artifact) return;
    
    const blob = new Blob([artifact.content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${artifact.title.replace(/\s+/g, "_")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Format content with proper markdown rendering
  const formattedContent = useMemo(() => {
    if (!artifact?.content) return null;

    // For code artifacts, wrap in code block
    if (artifact.type === "code") {
      return <CodeBlock language="python" value={artifact.content} />;
    }

    // For reports/documents, parse markdown with code blocks
    const parseCodeBlocks = (text: string) => {
      const segments = text.split(/(```[\s\S]*?```)/);
      return segments.map((segment, index) => {
        if (segment.startsWith("```") && segment.endsWith("```")) {
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
          return (
            <ReactMarkdown key={`text-${index}`}>
              {segment}
            </ReactMarkdown>
          );
        }

        return null;
      }).filter(Boolean);
    };

    return parseCodeBlocks(artifact.content);
  }, [artifact]);

  // Reset expansion when new artifact is shown
  useEffect(() => {
    setIsExpanded(true);
  }, [artifact?.id]);

  if (!artifact) return null;

  return (
    <div
      className={`flex h-full flex-col overflow-hidden border-l ${
        darkMode
          ? "border-gray-700 bg-gray-900"
          : "border-gray-200 bg-white"
      }`}
    >
      {/* Header */}
      <div
        className={`flex items-center justify-between border-b px-4 py-3 ${
          darkMode ? "border-gray-700 bg-gray-800" : "border-gray-200 bg-gray-50"
        }`}
      >
        <div className="flex items-center gap-2">
          <span className={getArtifactColor(artifact.type, darkMode)}>
            {getArtifactIcon(artifact.type)}
          </span>
          <h3
            className={`font-medium ${
              darkMode ? "text-white" : "text-gray-900"
            }`}
          >
            {artifact.title}
          </h3>
        </div>
        
        <div className="flex items-center gap-1">
          {/* Copy button */}
          <button
            onClick={handleCopy}
            className={`rounded-md p-1.5 transition-colors ${
              darkMode
                ? "text-gray-400 hover:bg-gray-700 hover:text-white"
                : "text-gray-500 hover:bg-gray-100 hover:text-gray-700"
            }`}
            title="Copy to clipboard"
          >
            {copied ? (
              <Check className="h-4 w-4 text-green-500" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </button>
          
          {/* Download button */}
          <button
            onClick={handleDownload}
            className={`rounded-md p-1.5 transition-colors ${
              darkMode
                ? "text-gray-400 hover:bg-gray-700 hover:text-white"
                : "text-gray-500 hover:bg-gray-100 hover:text-gray-700"
            }`}
            title="Download as file"
          >
            <Download className="h-4 w-4" />
          </button>
          
          {/* Expand/collapse button */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={`rounded-md p-1.5 transition-colors ${
              darkMode
                ? "text-gray-400 hover:bg-gray-700 hover:text-white"
                : "text-gray-500 hover:bg-gray-100 hover:text-gray-700"
            }`}
            title={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronUp className="h-4 w-4" />
            )}
          </button>
          
          {/* Close button */}
          <button
            onClick={onClose}
            className={`rounded-md p-1.5 transition-colors ${
              darkMode
                ? "text-gray-400 hover:bg-gray-700 hover:text-white"
                : "text-gray-500 hover:bg-gray-100 hover:text-gray-700"
            }`}
            title="Close panel"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Content area */}
      {isExpanded && (
        <div
          className={`flex-1 overflow-y-auto p-4 ${
            darkMode ? "text-gray-200" : "text-gray-800"
          }`}
        >
          <div className="prose prose-sm max-w-none dark:prose-invert">
            {formattedContent}
          </div>
        </div>
      )}

      {/* Footer with metadata */}
      <div
        className={`flex items-center justify-between border-t px-4 py-2 text-xs ${
          darkMode
            ? "border-gray-700 bg-gray-800 text-gray-500"
            : "border-gray-200 bg-gray-50 text-gray-400"
        }`}
      >
        <span className="capitalize">{artifact.type}</span>
        {artifact.timestamp && (
          <span>
            {new Date(artifact.timestamp).toLocaleString()}
          </span>
        )}
      </div>
    </div>
  );
};

export default ArtifactPanel;
