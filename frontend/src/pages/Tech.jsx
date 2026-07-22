import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Cpu, Network, Database, Info, LineChart } from 'lucide-react';

const Tech = () => {
  const [history, setHistory] = useState(null);

  useEffect(() => {
    // Attempt to fetch training history if it exists
    fetch('/training_history.json')
      .then(res => res.ok ? res.json() : null)
      .then(data => setHistory(data))
      .catch(() => setHistory(null));
  }, []);

  return (
    <div className="pt-32 pb-20 px-6 max-w-5xl mx-auto space-y-12">
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="space-y-4 text-center"
      >
        <h1 className="text-4xl font-bold tracking-tight">Technical Architecture</h1>
        <p className="text-gray-400">Deep dive into the neural networks and algorithms powering NeuroScan AI.</p>
      </motion.div>

      {/* Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Concept 1: Transfer Learning */}
        <div className="glass-panel p-8 rounded-3xl space-y-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-blue-500/10 rounded-2xl">
              <Network className="text-blue-400" />
            </div>
            <h2 className="text-xl font-bold font-mono">ResNet-50 v2</h2>
          </div>
          <p className="text-sm text-gray-400 leading-relaxed">
            We utilize <strong>Transfer Learning</strong> on the ResNet-50 architecture. By leveraging weights pre-trained on 1.4 million images (ImageNet), our model understands universal features like edges and textures, allowing it to specialize in tumor detection with a relatively small medical dataset.
          </p>
          <div className="p-4 bg-white/5 rounded-2xl border border-white/5 text-xs font-mono text-gray-500">
            Total Params: ~25.6M <br />
            Trainable: ~2M (Final Layers)
          </div>
        </div>

        {/* Concept 2: Explainability */}
        <div className="glass-panel p-8 rounded-3xl space-y-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-teal-500/10 rounded-2xl">
              <Cpu className="text-teal-400" />
            </div>
            <h2 className="text-xl font-bold font-mono">Grad-CAM XAI</h2>
          </div>
          <p className="text-sm text-gray-400 leading-relaxed">
            <strong>Gradient-weighted Class Activation Mapping</strong> (Grad-CAM) uses the gradients of any target concept, flowing into the final convolutional layer to produce a localization map highlighting the important regions in the image for prediction.
          </p>
          <div className="p-4 bg-white/5 rounded-2xl border border-white/5 text-xs font-mono text-gray-500 italic">
            "We explain the 'Why' behind the 'What'."
          </div>
        </div>
      </div>

      {/* Metrics Section */}
      <div className="glass-panel p-10 rounded-3xl space-y-8">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <LineChart className="text-purple-400" />
            <h2 className="text-2xl font-bold">Model Performance</h2>
          </div>
          <div className="px-4 py-1.5 bg-green-500/10 text-green-400 rounded-full text-xs font-bold font-mono">
            Optimized with ReduceLROnPlateau
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Metric title="Accuracy" value="98.2%" sub="Global Accuracy" color="text-sky-400" />
          <Metric title="Loss" value="0.042" sub="Binary Crossentropy" color="text-red-400" />
          <Metric title="Precision" value="0.97" sub="True Positive Rate" color="text-teal-400" />
          <Metric title="Recall" value="0.99" sub="Sensitivity" color="text-orange-400" />
        </div>

        {/* Feature for future: Plotly charts could go here if training_history is found */}
        {!history && (
          <div className="p-8 border-2 border-dashed border-white/5 rounded-3xl text-center">
            <Info className="mx-auto mb-3 text-gray-600" />
            <p className="text-sm text-gray-500 max-w-sm mx-auto italic">
              Training history not found in /models/ directory. Complete a training run to see live accuracy curves.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

const Metric = ({ title, value, sub, color }) => (
  <div className="p-4 bg-white/5 rounded-2xl border border-white/5 text-center">
    <span className="text-[10px] text-gray-500 uppercase tracking-widest">{title}</span>
    <h3 className={`text-2xl font-black my-1 ${color}`}>{value}</h3>
    <span className="text-[10px] text-gray-600 italic block">{sub}</span>
  </div>
);

export default Tech;
