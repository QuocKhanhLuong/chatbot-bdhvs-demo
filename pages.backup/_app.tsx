import "@/styles/globals.css";
import type { AppProps } from "next/app";
import { Inter } from "next/font/google";
import { DarkModeProvider } from "@/components/Contexts/DarkModeContext";

const inter = Inter({ subsets: ["latin"] });

export default function App({ Component, pageProps }: AppProps<{}>) {
  return (
    <DarkModeProvider>
      <main className={`${inter.className} min-h-screen bg-background`}>
        <Component {...pageProps} />
      </main>
    </DarkModeProvider>
  );
}
