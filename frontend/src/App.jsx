import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Diagnosis from './pages/Diagnosis';
import Tech from './pages/Tech';

const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-[#05070a] text-white">
        {/* Animated Background Orbs */}
        <div className="glow-orb bg-sky-500 w-[600px] h-[600px] -top-40 -left-40 opacity-10" />
        <div className="glow-orb bg-teal-500 w-[500px] h-[500px] bottom-0 -right-40 opacity-10" />
        <div className="glow-orb bg-purple-500 w-[400px] h-[400px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-5" />

        <Navbar />
        
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analysis" element={<Diagnosis />} />
          <Route path="/tech" element={<Tech />} />
        </Routes>

        {/* Global Footer */}
        <footer className="w-full py-12 px-6 flex flex-col items-center justify-center border-t border-white/5 bg-black/20">
          <p className="text-[10px] text-gray-500 text-center max-w-xl uppercase tracking-[0.2em] leading-relaxed">
            NeuroScan AI Diagnostic Research Tool <br />
            Educational Purpose Only • Not for Clinical Decisions • © 2026
          </p>
        </footer>
      </div>
    </Router>
  );
};

export default App;
