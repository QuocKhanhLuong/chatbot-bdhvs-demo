import { GoogleGenerativeAI } from "@google/generative-ai";

const apiKey = process.env.GEMINI_API_KEY;

if (!apiKey) {
  throw new Error("GEMINI_API_KEY is not configured in environment variables");
}

const genAI = new GoogleGenerativeAI(apiKey);

const model = genAI.getGenerativeModel({
  model: "gemini-2.0-flash-lite",
  generationConfig: {
    temperature: 0.7,
    maxOutputTokens: 2048,
  }
});

// Default role configuration
const defaultRole = {
  role: "Trợ lý ảo hỗ trợ Hệ thống thi Bình dân học vụ số",
  expertise: "Giải đáp quy chế thi, hướng dẫn xử lý sự cố kỹ thuật, tra cứu điểm thi",
  tone: "Chuyên nghiệp, ngắn gọn, chính xác, thân thiện",
  language: "Vietnamese"
};

// Helper function to delay execution
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Hàm giả lập streaming bằng cách chia nhỏ text
async function simulateStreaming(text, onStream, delayMs = 50) {
  const sentences = text.split(/([.!?]+)/);
  let fullText = '';

  for (let i = 0; i < sentences.length; i += 2) {
    const sentence = sentences[i] + (sentences[i + 1] || '');
    fullText += sentence;
    onStream(sentence);
    await delay(delayMs);
  }

  return fullText;
}

async function generateResponse(prompt, context = "", roleConfig = defaultRole, onStream = null) {
  const maxRetries = 3;
  const retryDelays = [1000, 2000, 4000]; // Exponential backoff: 1s, 2s, 4s

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      console.log(`Starting Gemini API call (attempt ${attempt + 1}/${maxRetries + 1})`);

      // Construct the system prompt with role information and context
      let systemPrompt = `Bạn là Trợ lý ảo của Hệ thống thi Bình dân học vụ số với các đặc điểm sau:
- Vai trò: ${roleConfig.role}
- Chuyên môn: ${roleConfig.expertise}
- Phong cách: ${roleConfig.tone}
- Ngôn ngữ: ${roleConfig.language}`;

      // Add context section if provided
      if (context && context.trim()) {
        systemPrompt += `

### TÀI LIỆU THAM KHẢO (CONTEXT):
${context}

HƯỚNG DẪN QUAN TRỌNG:
- Hãy ưu tiên sử dụng thông tin trong phần 'TÀI LIỆU THAM KHẢO' để trả lời câu hỏi.
- Nếu thông tin không có trong tài liệu tham khảo, hãy lịch sự nói rằng bạn không tìm thấy thông tin và gợi ý thí sinh liên hệ hotline 1900-xxxx để được hỗ trợ trực tiếp.
- TUYỆT ĐỐI KHÔNG tự bịa đặt hoặc suy luận các quy chế, quy định không có trong tài liệu.`;
      }

      systemPrompt += `

Lưu ý: Bạn phải trả lời bằng tiếng Việt. Dù người dùng nói ngôn ngữ nào, bạn cũng phải trả lời bằng tiếng Việt. Không cần dịch tin nhắn của người dùng, chỉ cần trả lời bằng tiếng Việt.

Hãy trả lời câu hỏi sau của người dùng trong khi duy trì vai trò và phong cách đã nêu:
${prompt}`;

      if (onStream) {
        // Sử dụng chế độ giả lập streaming
        const result = await model.generateContent(systemPrompt);
        const response = await result.response;
        const fullText = response.text();

        return await simulateStreaming(fullText, onStream);
      } else {
        // Non-streaming mode
        const result = await model.generateContent(systemPrompt);
        const response = await result.response;
        return response.text();
      }

    } catch (error) {
      console.error(`Error in Gemini API (attempt ${attempt + 1}):`, error);

      // Check if it's a 429 rate limit error
      const is429Error = error.message?.includes('429') ||
        error.status === 429 ||
        error.message?.toLowerCase().includes('rate limit') ||
        error.message?.toLowerCase().includes('quota');

      // If it's a 429 error and we have retries left, wait and retry
      if (is429Error && attempt < maxRetries) {
        const delayMs = retryDelays[attempt];
        console.log(`Rate limit hit. Retrying after ${delayMs}ms...`);
        await delay(delayMs);
        continue; // Retry
      }

      // If we've exhausted retries or it's a different error, return Vietnamese error message
      if (is429Error) {
        return "Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau ít phút.";
      } else {
        return "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại sau.";
      }
    }
  }

  // This should never be reached, but just in case
  return "Xin lỗi, hệ thống đang quá tải. Vui lòng thử lại sau ít phút.";
}

// Export as named export
export { generateResponse };

// Keep default export for backward compatibility
export default generateResponse;