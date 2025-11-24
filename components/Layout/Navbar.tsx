import { FC } from "react";
import { GraduationCap } from "lucide-react";

export const Navbar: FC = () => {
  return (
    <nav className="bg-primary border-b border-primary-dark">
      <div className="max-w-[800px] mx-auto px-4 py-4 flex items-center justify-between">
        {/* Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-sm">
            <GraduationCap className="text-white" size={24} />
          </div>
          <h1 className="text-xl font-bold text-white">
            Hệ thống Thi Bình dân học vụ số
          </h1>
        </div>
      </div>
    </nav>
  );
};
