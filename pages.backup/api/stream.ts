import { Message } from "@/types";
import type { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Set headers for SSE
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // Disable buffering for Nginx

  try {
    const { messages, streamId } = req.body as {
      messages: Message[];
      streamId?: string;
    };

    if (!messages || !Array.isArray(messages)) {
      res.write(`data: ${JSON.stringify({ error: 'Messages are required and must be an array' })}\\n\\n`);
      res.end();
      return;
    }

    // Get the last message from user
    const lastMessage = messages[messages.length - 1];
    console.log("Processing message for streaming:", lastMessage);

    if (!lastMessage || !lastMessage.content) {
      res.write(`data: ${JSON.stringify({ error: 'Invalid message format' })}\\n\\n`);
      res.end();
      return;
    }

    // Send an initial message to establish the connection
    res.write(`data: ${JSON.stringify({ type: 'start', streamId })}\\n\\n`);

    try {
      console.log(`Processing message with RAG:`, lastMessage.content);

      // Import our local AI and RAG functions (handle both ESM and CommonJS)
      const geminiModule = await import('../../gemini.js');
      const ragModule = await import('../../utils/rag');
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

        res.write(`data: ${JSON.stringify({
          type: 'chunk',
          content: accumulatedContent,
          streamId
        })}\\n\\n`);

        // Add delay for typing effect
        await new Promise(resolve => setTimeout(resolve, 20));
      }

      // Send completion message
      res.write(`data: ${JSON.stringify({
        type: 'end',
        content: accumulatedContent,
        streamId
      })}\\n\\n`);

      res.end();
    } catch (error: any) {
      console.error("Streaming error:", error);
      res.write(`data: ${JSON.stringify({
        error: error.message || 'Lỗi khi xử lý yêu cầu',
        streamId
      })}\\n\\n`);
      res.end();
    }
  } catch (error: any) {
    console.error("Error setting up SSE:", error);
    res.write(`data: ${JSON.stringify({ error: error.message || 'Lỗi server' })}\\n\\n`);
    res.end();
  }
}