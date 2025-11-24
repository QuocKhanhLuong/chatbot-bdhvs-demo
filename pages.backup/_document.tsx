import { DarkModeProvider } from '@/components/Contexts/DarkModeContext'
import { Html, Head, Main, NextScript } from 'next/document'

export default function Document() {
  return (
    <DarkModeProvider>
      <Html lang="en">
        <Head />
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    </DarkModeProvider> 
  )
}
