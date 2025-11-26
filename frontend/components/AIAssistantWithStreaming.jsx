"use client"

import React, { useEffect, useMemo, useRef, useState, useCallback } from "react"
import { Calendar, LayoutGrid, MoreHorizontal } from "lucide-react"
import { useTheme } from "next-themes"
import Sidebar from "./Sidebar"
import Header from "./Header"
import ChatPane from "./ChatPane"
import GhostIconButton from "./GhostIconButton"
import ThemeToggle from "./ThemeToggle"
import { INITIAL_FOLDERS, INITIAL_TEMPLATES } from "./mockData"

export default function AIAssistantWithStreaming() {
  const { theme, setTheme } = useTheme()

  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(() => {
    try {
      const raw = localStorage.getItem("sidebar-collapsed")
      return raw ? JSON.parse(raw) : { pinned: true, recent: false, folders: true, templates: true }
    } catch {
      return { pinned: true, recent: false, folders: true, templates: true }
    }
  })
  
  useEffect(() => {
    try {
      localStorage.setItem("sidebar-collapsed", JSON.stringify(collapsed))
    } catch {}
  }, [collapsed])

  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try {
      const saved = localStorage.getItem("sidebar-collapsed-state")
      return saved ? JSON.parse(saved) : false
    } catch {
      return false
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem("sidebar-collapsed-state", JSON.stringify(sidebarCollapsed))
    } catch {}
  }, [sidebarCollapsed])

  // Conversations state - each conversation has messages
  const [conversations, setConversations] = useState(() => {
    const initialConv = {
      id: "initial",
      title: "New Conversation",
      updatedAt: new Date().toISOString(),
      messageCount: 1,
      preview: "Xin chào! Tôi là Chatbot Soni...",
      pinned: false,
      folder: "Personal",
      messages: [
        {
          id: "msg-1",
          role: "assistant",
          content: "Xin chào! Tôi là Chatbot Soni, một trợ lý AI. Tôi có thể giúp bạn với những việc như trả lời câu hỏi, cung cấp thông tin, và hỗ trợ các nhiệm vụ. Tôi có thể giúp gì cho bạn?",
          createdAt: new Date().toISOString(),
        },
      ],
    }
    return [initialConv]
  })
  
  const [selectedId, setSelectedId] = useState("initial")
  const [templates, setTemplates] = useState(INITIAL_TEMPLATES)
  const [folders, setFolders] = useState(INITIAL_FOLDERS)

  const [query, setQuery] = useState("")
  const searchRef = useRef(null)

  const [isThinking, setIsThinking] = useState(false)
  const [thinkingConvId, setThinkingConvId] = useState(null)
  
  // Streaming state
  const abortControllerRef = useRef(null)
  const streamIdRef = useRef("")

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") {
        e.preventDefault()
        createNewChat()
      }
      if (!e.metaKey && !e.ctrlKey && e.key === "/") {
        const tag = document.activeElement?.tagName?.toLowerCase()
        if (tag !== "input" && tag !== "textarea") {
          e.preventDefault()
          searchRef.current?.focus()
        }
      }
      if (e.key === "Escape" && sidebarOpen) setSidebarOpen(false)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [sidebarOpen])

  const filtered = useMemo(() => {
    if (!query.trim()) return conversations
    const q = query.toLowerCase()
    return conversations.filter((c) => c.title.toLowerCase().includes(q) || c.preview.toLowerCase().includes(q))
  }, [conversations, query])

  const pinned = filtered.filter((c) => c.pinned).sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1))

  const recent = filtered
    .filter((c) => !c.pinned)
    .sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1))
    .slice(0, 10)

  const folderCounts = React.useMemo(() => {
    const map = Object.fromEntries(folders.map((f) => [f.name, 0]))
    for (const c of conversations) if (map[c.folder] != null) map[c.folder] += 1
    return map
  }, [conversations, folders])

  function togglePin(id) {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c)))
  }

  function createNewChat() {
    const id = Math.random().toString(36).slice(2)
    const item = {
      id,
      title: "New Conversation",
      updatedAt: new Date().toISOString(),
      messageCount: 1,
      preview: "Say hello to start...",
      pinned: false,
      folder: "Personal",
      messages: [
        {
          id: "msg-initial",
          role: "assistant",
          content: "Xin chào! Tôi là Chatbot Soni. Tôi có thể giúp gì cho bạn?",
          createdAt: new Date().toISOString(),
        },
      ],
    }
    setConversations((prev) => [item, ...prev])
    setSelectedId(id)
    setSidebarOpen(false)
  }

  function createFolder() {
    const name = prompt("Folder name")
    if (!name) return
    if (folders.some((f) => f.name.toLowerCase() === name.toLowerCase())) return alert("Folder already exists.")
    setFolders((prev) => [...prev, { id: Math.random().toString(36).slice(2), name }])
  }

  // Clean up any active connections
  const cleanupConnections = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    streamIdRef.current = ""
  }, [])

  async function sendMessage(convId, content) {
    if (!content.trim()) return
    
    const now = new Date().toISOString()
    const userMsg = { id: Math.random().toString(36).slice(2), role: "user", content, createdAt: now }

    // Add user message
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c
        const msgs = [...(c.messages || []), userMsg]
        return {
          ...c,
          messages: msgs,
          updatedAt: now,
          messageCount: msgs.length,
          preview: content.slice(0, 80),
          title: c.messageCount === 1 ? content.slice(0, 50) : c.title,
        }
      }),
    )

    setIsThinking(true)
    setThinkingConvId(convId)

    // Clean up any existing connections
    cleanupConnections()

    try {
      // Create placeholder for assistant response
      const assistantMsgId = Math.random().toString(36).slice(2)
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c
          const msgs = [
            ...(c.messages || []),
            {
              id: assistantMsgId,
              role: "assistant",
              content: "",
              isStreaming: true,
              createdAt: new Date().toISOString(),
            },
          ]
          return { ...c, messages: msgs, messageCount: msgs.length }
        }),
      )

      // Generate a unique ID for this streaming session
      const streamId = Date.now().toString()
      streamIdRef.current = streamId

      // Create controller for this request
      const controller = new AbortController()
      abortControllerRef.current = controller

      // Get all messages for context
      const conv = conversations.find((c) => c.id === convId)
      const allMessages = [...(conv?.messages || []), userMsg]

      // Make the stream request
      const response = await fetch('/api/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          messages: allMessages,
          streamId: streamId
        }),
        signal: controller.signal
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error("Failed to get response reader")
      }

      // Read the stream
      const decoder = new TextDecoder()
      let accumulatedChunks = ''

      while (true) {
        const { done, value } = await reader.read()

        if (done) {
          // Stream complete
          setIsThinking(false)
          setThinkingConvId(null)

          // Finalize the message
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c
              const msgs = (c.messages || []).map((m) =>
                m.id === assistantMsgId
                  ? { ...m, content: accumulatedChunks, isStreaming: false }
                  : m
              )
              return {
                ...c,
                messages: msgs,
                preview: accumulatedChunks.slice(0, 80),
              }
            }),
          )

          break
        }

        // Decode and process the chunk
        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const eventData = JSON.parse(line.slice(6))

              if (eventData.error) {
                console.error("Stream error:", eventData.error)
                setIsThinking(false)
                setThinkingConvId(null)
                return
              }

              if (eventData.type === 'chunk') {
                accumulatedChunks = eventData.content
                
                // Update the streaming message
                setConversations((prev) =>
                  prev.map((c) => {
                    if (c.id !== convId) return c
                    const msgs = (c.messages || []).map((m) =>
                      m.id === assistantMsgId ? { ...m, content: accumulatedChunks } : m
                    )
                    return { ...c, messages: msgs }
                  }),
                )
              }
              
              // Handle end event with agent info
              if (eventData.type === 'end') {
                accumulatedChunks = eventData.content
                const agentUsed = eventData.agentUsed || 'general'
                
                // Final update with agent info
                setConversations((prev) =>
                  prev.map((c) => {
                    if (c.id !== convId) return c
                    const msgs = (c.messages || []).map((m) =>
                      m.id === assistantMsgId 
                        ? { ...m, content: accumulatedChunks, agentUsed: agentUsed, isStreaming: false } 
                        : m
                    )
                    return { ...c, messages: msgs, preview: accumulatedChunks.slice(0, 80) }
                  }),
                )
              }
            } catch (e) {
              console.error("Error parsing SSE message:", e, line)
            }
          }
        }
      }

      reader.releaseLock()
    } catch (error) {
      if (controller.signal.aborted) {
        console.log("Stream aborted")
      } else {
        console.error("Error:", error)
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convId) return c
            const msgs = [
              ...(c.messages || []),
              {
                id: Math.random().toString(36).slice(2),
                role: "assistant",
                content: "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
                createdAt: new Date().toISOString(),
              },
            ]
            return { ...c, messages: msgs, messageCount: msgs.length }
          }),
        )
      }
      setIsThinking(false)
      setThinkingConvId(null)
    }
  }

  function editMessage(convId, messageId, newContent) {
    const now = new Date().toISOString()
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c
        const msgs = (c.messages || []).map((m) =>
          m.id === messageId ? { ...m, content: newContent, editedAt: now } : m,
        )
        return {
          ...c,
          messages: msgs,
          preview: msgs[msgs.length - 1]?.content?.slice(0, 80) || c.preview,
        }
      }),
    )
  }

  function resendMessage(convId, messageId) {
    const conv = conversations.find((c) => c.id === convId)
    const msg = conv?.messages?.find((m) => m.id === messageId)
    if (!msg) return
    sendMessage(convId, msg.content)
  }

  function pauseThinking() {
    cleanupConnections()
    setIsThinking(false)
    setThinkingConvId(null)
  }

  function handleUseTemplate(template) {
    if (composerRef.current) {
      composerRef.current.insertTemplate(template.content)
    }
  }

  const composerRef = useRef(null)
  const selected = conversations.find((c) => c.id === selectedId) || null

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupConnections()
    }
  }, [cleanupConnections])

  return (
    <div className="h-screen w-full bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      <div className="md:hidden sticky top-0 z-40 flex items-center gap-2 border-b border-zinc-200/60 bg-white/80 px-3 py-2 backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/70">
        <div className="ml-1 flex items-center gap-2 text-sm font-semibold tracking-tight">
          <span className="inline-flex h-4 w-4 items-center justify-center">✱</span> Chatbot Soni
        </div>
        <div className="ml-auto flex items-center gap-2">
          <GhostIconButton label="Schedule">
            <Calendar className="h-4 w-4" />
          </GhostIconButton>
          <GhostIconButton label="Apps">
            <LayoutGrid className="h-4 w-4" />
          </GhostIconButton>
          <GhostIconButton label="More">
            <MoreHorizontal className="h-4 w-4" />
          </GhostIconButton>
          <ThemeToggle theme={theme} setTheme={setTheme} />
        </div>
      </div>

      <div className="mx-auto flex h-[calc(100vh-0px)] max-w-[1400px]">
        <Sidebar
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          theme={theme}
          setTheme={setTheme}
          collapsed={collapsed}
          setCollapsed={setCollapsed}
          sidebarCollapsed={sidebarCollapsed}
          setSidebarCollapsed={setSidebarCollapsed}
          conversations={conversations}
          pinned={pinned}
          recent={recent}
          folders={folders}
          folderCounts={folderCounts}
          selectedId={selectedId}
          onSelect={(id) => setSelectedId(id)}
          togglePin={togglePin}
          query={query}
          setQuery={setQuery}
          searchRef={searchRef}
          createFolder={createFolder}
          createNewChat={createNewChat}
          templates={templates}
          setTemplates={setTemplates}
          onUseTemplate={handleUseTemplate}
        />

        <main className="relative flex min-w-0 flex-1 flex-col">
          <Header createNewChat={createNewChat} sidebarCollapsed={sidebarCollapsed} setSidebarOpen={setSidebarOpen} />
          <ChatPane
            ref={composerRef}
            conversation={selected}
            onSend={(content) => selected && sendMessage(selected.id, content)}
            onEditMessage={(messageId, newContent) => selected && editMessage(selected.id, messageId, newContent)}
            onResendMessage={(messageId) => selected && resendMessage(selected.id, messageId)}
            isThinking={isThinking && thinkingConvId === selected?.id}
            onPauseThinking={pauseThinking}
          />
        </main>
      </div>
    </div>
  )
}
