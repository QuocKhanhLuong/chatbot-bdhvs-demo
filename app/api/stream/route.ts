import { NextRequest } from 'next/server'
import { Message } from "@/types"

export const runtime = 'nodejs' // or 'edge' if you prefer edge runtime
export const dynamic = 'force-dynamic' // disable caching

export async function POST(request: NextRequest) {
  try {
    const { messages, streamId } = await request.json() as {
      messages: Message[];
      streamId?: string;
    };

    if (!messages || !Array.isArray(messages)) {
      return new Response(
        JSON.stringify({ error: 'Messages are required and must be an array' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Get the last message from user
    const lastMessage = messages[messages.length - 1];
    console.log("Processing message for streaming:", lastMessage);

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
            encoder.encode(`data: ${JSON.stringify(data)}\\n\\n`)
          );
        };

        try {
          // Send start message
          sendSSE({ type: 'start', streamId });

          console.log(`Processing message with RAG:`, lastMessage.content);

          // Import our local AI and RAG functions
          const geminiModule = await import('../../../gemini.js');
          const ragModule = await import('../../../utils/rag');
          const generateResponse = geminiModule.default || geminiModule.generateResponse || geminiModule;
          const retrieveContext = ragModule.retrieveContext;

          // Retrieve context from knowledge base
          const context = await retrieveContext(lastMessage.content);
          console.log("Retrieved context for streaming:", context ? "Context found" : "No context");

          // Generate response with context
          const fullResponse = await generateResponse(lastMessage.content, context);

          // Simulate streaming by sending characters one by one
          let accumulatedContent = '';
          const characters = fullResponse.split('');

          for (const char of characters) {
            accumulatedContent += char;

            sendSSE({
              type: 'chunk',
              content: accumulatedContent,
              streamId
            });

            // Add delay for typing effect
            await new Promise(resolve => setTimeout(resolve, 20));
          }

          // Send completion message
          sendSSE({
            type: 'end',
            content: accumulatedContent,
            streamId
          });

          controller.close();
        } catch (error: any) {
          console.error("Streaming error:", error);
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
