import { FC } from "react";
import { useDarkMode } from "../Contexts/DarkModeContext";
import { RotateCcw } from "lucide-react";

interface Props {
  onReset: () => void;
}

export const ResetChat: FC<Props> = ({ onReset }) => {
  const { darkMode } = useDarkMode();
  
  return (
    <div className="flex flex-row items-center">
      <button
        className={`inline-flex items-center gap-2 text-sm sm:text-base ${
          darkMode 
            ? 'text-white bg-gray-700 hover:bg-gray-600 focus:ring-pink-500 border-gray-600' 
            : 'text-gray-700 bg-white hover:bg-pink-50 focus:ring-pink-500 border-pink-200'
        } font-medium rounded-lg px-4 py-2 border focus:outline-none focus:ring-2 transition-all`}
        onClick={() => onReset()}
      >
        <RotateCcw className="h-4 w-4" />
        Reset
      </button>
    </div>
  );
};