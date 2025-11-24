import { FC } from "react";

interface Props {}

export const ChatLoader: FC<Props> = () => {
  return (
    <div className="flex gap-3 justify-start">
      <div className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-gradient-to-br from-pink-500 to-rose-500 text-xs font-bold text-white shadow-sm">
        AI
      </div>
      <div
        className="flex items-center gap-2 rounded-2xl border border-pink-100 bg-white px-4 py-2 shadow-sm dark:border-gray-700 dark:bg-gray-800"
        style={{ overflowWrap: "anywhere" }}
      >
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 animate-bounce rounded-full bg-pink-400 [animation-delay:-0.3s]"></div>
          <div className="h-2 w-2 animate-bounce rounded-full bg-pink-400 [animation-delay:-0.15s]"></div>
          <div className="h-2 w-2 animate-bounce rounded-full bg-pink-400"></div>
        </div>
        <span className="text-sm text-gray-500 dark:text-gray-400">AI is thinking...</span>
      </div>
    </div>
  );
};
