import { NextRequest } from 'next/server'
import { Message } from "@/types"

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

// Python FastAPI backend URL
const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000'

interface Source {
  content: string;
  page: number;
  source: string;
}

interface BackendResponse {
  response: string;
  thread_id: string;
  tool_calls?: any[];
  agent_used?: string;
}

export async function POST(request: NextRequest) {
  try {
    const { messages, streamId, includeSources = true } = await request.json() as {
      messages: Message[];
      streamId?: string;
      includeSources?: boolean;
    };

    if (!messages || !Array.isArray(messages)) {
      return new Response(
        JSON.stringify({ error: 'Messages are required and must be an array' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Get the last message from user
    const lastMessage = messages[messages.length - 1];
    console.log("📨 Forwarding to Python backend:", lastMessage.content.substring(0, 100));

    if (!lastMessage || !lastMessage.content) {
      return new Response(
        JSON.stringify({ error: 'Invalid message format' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Create a ReadableStream for SSE
    const encoder = new TextEncoder();
    
    const stream = new ReadableStream({
      async start(controller) {
        // Helper to send SSE messages
        const sendSSE = (data: any) => {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(data)}\n\n`)
          );
        };

        try {
          // Send start message
          sendSSE({ type: 'start', streamId });

          console.log(`🔗 Calling Python backend at ${PYTHON_BACKEND_URL}/agent/chat`);

          // Call Python FastAPI backend (multi-agent endpoint)
          const response = await fetch(`${PYTHON_BACKEND_URL}/agent/chat`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              message: lastMessage.content,
              thread_id: streamId
            })
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Backend error: ${response.status}`);
          }

          const data: BackendResponse = await response.json();
          const fullResponse = data.response;
          const agentUsed = data.agent_used || 'general';

          console.log(`✅ Received response from Python backend (${fullResponse.length} chars, agent: ${agentUsed})`);

          // Simulate streaming by sending characters one by one for smooth UX
          let accumulatedContent = '';
          const characters = fullResponse.split('');

          for (const char of characters) {
            accumulatedContent += char;

            sendSSE({
              type: 'chunk',
              content: accumulatedContent,
              streamId
            });

            // Add small delay for typing effect
            await new Promise(resolve => setTimeout(resolve, 15));
          }

          // Send completion message
          sendSSE({
            type: 'end',
            content: accumulatedContent,
            agentUsed: agentUsed,
            streamId
          });

          controller.close();
        } catch (error: any) {
          console.error("❌ Streaming error:", error);
          sendSSE({
            error: error.message || 'Lỗi khi xử lý yêu cầu',
            streamId
          });
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (error: any) {
    console.error("Error setting up SSE:", error);
    return new Response(
      JSON.stringify({ error: error.message || 'Lỗi server' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
