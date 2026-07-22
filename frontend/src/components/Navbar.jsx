import React from 'react';
import { NavLink } from 'react-router-dom';
import { Brain, Search, Cpu, Home } from 'lucide-react';

const Navbar = () => {
  return (
    <nav className="fixed top-6 left-1/2 -translate-x-1/2 z-50 w-[90%] max-w-2xl glass-panel rounded-2xl px-6 py-4">
      <div className="flex items-center justify-between">
        <NavLink to="/" className="flex items-center gap-2 group">
          <div className="p-1.5 bg-sky-500/20 rounded-lg group-hover:bg-sky-500/30 transition-colors">
            <Brain className="text-sky-400 w-5 h-5" />
          </div>
          <span className="font-bold tracking-tight hidden sm:block">NeuroScan <span className="text-sky-400">AI</span></span>
        </NavLink>

        <div className="flex items-center gap-1 sm:gap-4">
          <NavButton to="/" icon={<Home size={18} />} label="Home" />
          <NavButton to="/analysis" icon={<Search size={18} />} label="Analysis" />
          <NavButton to="/tech" icon={<Cpu size={18} />} label="Tech" />
        </div>
      </div>
    </nav>
  );
};

const NavButton = ({ to, icon, label }) => (
  <NavLink
    to={to}
    className={({ isActive }) => `
      flex items-center gap-2 px-4 py-2 rounded-xl transition-all text-sm font-medium
      ${isActive ? 'bg-white/10 text-sky-400 shadow-inner' : 'text-gray-400 hover:text-white hover:bg-white/5'}
    `}
  >
    {icon}
    <span className="hidden md:block">{label}</span>
  </NavLink>
);

export default Navbar;
