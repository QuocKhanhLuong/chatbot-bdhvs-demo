import AIAssistantWithStreaming from "@/components/AIAssistantWithStreaming";
import Head from "next/head";

export default function Home() {
  return (
    <>
      <Head>
        <title>Chatbot Soni - AI Assistant</title>
        <meta name="description" content="AI Chatbot với RAG và streaming" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/Sinno_logo.png" />
      </Head>

      <AIAssistantWithStreaming />
    </>
  );
}

