import { Message } from "@/types";
import type { NextApiRequest, NextApiResponse } from 'next';
import { generateResponse } from '../../gemini.js';
import { retrieveContext } from '../../utils/rag';

// In-memory cache for responses
const responseCache = new Map<string, string>();

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  try {
    if (req.method !== 'POST') {
      return res.status(405).json({ error: 'Method not allowed' });
    }

    const { messages } = req.body as {
      messages: Message[];
    };

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Messages are required and must be an array' });
    }

    // Get the last message from user
    const lastMessage = messages[messages.length - 1];
    console.log("Processing message:", lastMessage);

    if (!lastMessage || !lastMessage.content) {
      throw new Error("Invalid message format");
    }

    // Use trimmed content as cache key
    const cacheKey = lastMessage.content.trim();

    // Check if we have a cached response
    if (responseCache.has(cacheKey)) {
      console.log("Cache hit! Returning cached response");
      const cachedContent = responseCache.get(cacheKey)!;
      return res.status(200).json({
        role: "assistant",
        content: cachedContent,
        streaming: true,
        cached: true
      });
    }

    // No cache, retrieve context from knowledge base
    console.log("Cache miss. Retrieving RAG context...");
    const context = await retrieveContext(lastMessage.content);
    console.log("Retrieved RAG Context:", context || "(no matching documents)");

    // Generate new response with context
    console.log("Calling Gemini API with context...");
    const content = await generateResponse(lastMessage.content, context);

    // Store in cache
    responseCache.set(cacheKey, content);
    console.log(`Response cached. Cache size: ${responseCache.size}`);

    // Return the response
    const jsonResponse = {
      role: "assistant",
      content: content,
      streaming: true,
      cached: false
    };
    console.log("Sending response");

    return res.status(200).json(jsonResponse);
  } catch (error: any) {
    console.error("Error in chat API:", error);
    return res.status(500).json({
      error: error.message || "Internal server error"
    });
  }
};

export default handler;
