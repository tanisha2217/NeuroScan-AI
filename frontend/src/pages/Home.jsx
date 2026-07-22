import React from 'react';
import { motion } from 'framer-motion';
import { Shield, Zap, Eye, BarChart3, ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="pt-32 pb-20 px-6 flex flex-col items-center max-w-6xl mx-auto">
      {/* Hero Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-6 mb-20"
      >
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-panel border-sky-500/30 text-sky-400 text-xs font-semibold tracking-widest uppercase">
          <Zap size={14} /> AI Powered Diagnostics
        </div>
        <h1 className="text-5xl md:text-7xl font-black tracking-tighter leading-none">
          Next Generation <br />
          <span className="bg-gradient-to-r from-sky-400 to-teal-400 bg-clip-text text-transparent">Brain Analysis</span>
        </h1>
        <p className="text-gray-400 max-w-2xl mx-auto text-lg leading-relaxed px-4">
          Experience the future of medical imaging. Our AI-driven platform identifies tumors with 98% accuracy and visualizes decision-making using Explainable AI.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
          <Link to="/analysis" className="bg-sky-600 hover:bg-sky-500 text-white font-bold py-4 px-10 rounded-2xl transition-all shadow-xl shadow-sky-600/20 flex items-center gap-2 group">
            Start Diagnostic <ChevronRight className="group-hover:translate-x-1 transition-transform" />
          </Link>
          <Link to="/tech" className="glass-panel hover:bg-white/5 text-white font-bold py-4 px-10 rounded-2xl transition-all">
            How it Works
          </Link>
        </div>
      </motion.div>

      {/* Feature Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
        <FeatureCard 
          icon={<Eye className="text-teal-400" />}
          title="Explainable AI"
          desc="We don't just give results. Our Grad-CAM heatmaps highlight exactly which area of the scan the AI focused on."
        />
        <FeatureCard 
          icon={<Shield className="text-sky-400" />}
          title="ResNet-50 Power"
          desc="Built on industry-standard residual networks, fine-tuned specifically for neural oncology patterns."
        />
        <FeatureCard 
          icon={<BarChart3 className="text-purple-400" />}
          title="Real-time Metrics"
          desc="Visualize model performance, confusion matrices, and accuracy trends directly in your browser."
        />
      </div>
    </div>
  );
};

const FeatureCard = ({ icon, title, desc }) => (
  <motion.div 
    whileHover={{ y: -5 }}
    className="glass-panel p-8 rounded-3xl space-y-4 text-left border-white/5"
  >
    <div className="p-3 bg-white/5 w-fit rounded-2xl">
      {icon}
    </div>
    <h3 className="text-xl font-bold">{title}</h3>
    <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
  </motion.div>
);

export default Home;
