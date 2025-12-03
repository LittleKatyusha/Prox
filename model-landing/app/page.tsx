'use client';

import { motion } from 'framer-motion';
import { Sparkles, Cpu, Zap, Star, Copy, Check } from 'lucide-react';
import { useState, useEffect } from 'react';

// Icon mapping based on model name patterns
const getIconForModel = (modelName: string) => {
  const name = modelName.toLowerCase();
  if (name.includes('opus-4-5') || name.includes('opus-4.5')) return Star;
  if (name.includes('opus-4-1') || name.includes('opus-4.1')) return Sparkles;
  if (name.includes('opus-4') || name.includes('opus 4')) return Cpu;
  if (name.includes('sonnet')) return Zap;
  if (name.includes('gemini')) return Sparkles;
  return Cpu; // default
};

// Color mapping based on model name patterns
const getColorForModel = (modelName: string) => {
  const name = modelName.toLowerCase();
  if (name.includes('opus-4-5') || name.includes('opus-4.5')) return 'from-purple-500 to-pink-500';
  if (name.includes('opus-4-1') || name.includes('opus-4.1')) return 'from-blue-500 to-cyan-500';
  if (name.includes('opus-4') || name.includes('opus 4')) return 'from-emerald-500 to-teal-500';
  if (name.includes('sonnet')) return 'from-orange-500 to-red-500';
  if (name.includes('gemini')) return 'from-indigo-500 to-purple-500';
  return 'from-gray-500 to-gray-700'; // default
};

interface Model {
  name: string;
  version: string;
  description: string;
  color: string;
  icon: any;
}

interface ModelsByVersion {
  [key: string]: Model[];
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

export default function Home() {
  const [copiedVersion, setCopiedVersion] = useState<string | null>(null);
  const [modelsByVersion, setModelsByVersion] = useState<ModelsByVersion>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/models');
      
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      
      const data = await response.json();
      const modelsData = data.models_by_version || {};
      
      // Transform the data to include icons and colors
      const transformedModels: ModelsByVersion = {};
      
      Object.keys(modelsData).forEach((versionKey) => {
        transformedModels[versionKey] = modelsData[versionKey].map((model: any) => ({
          name: model.name,
          version: model.version,
          description: model.description,
          color: getColorForModel(model.name),
          icon: getIconForModel(model.name)
        }));
      });
      
      setModelsByVersion(transformedModels);
      setError(null);
    } catch (err) {
      console.error('Error fetching models:', err);
      setError('Failed to load models. Please make sure the server is running.');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async (version: string) => {
    try {
      // Check if clipboard API is available (client-side only)
      if (typeof window !== 'undefined' && navigator.clipboard) {
        await navigator.clipboard.writeText(version);
        setCopiedVersion(version);
        setTimeout(() => setCopiedVersion(null), 2000);
      } else {
        // Fallback for older browsers or SSR
        const textArea = document.createElement('textarea');
        textArea.value = version;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        setCopiedVersion(version);
        setTimeout(() => setCopiedVersion(null), 2000);
      }
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
      {/* Animated background - only render on client */}
      {mounted && (
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -inset-[10px] opacity-50">
            {[...Array(50)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute h-1 w-1 bg-white rounded-full"
                initial={{
                  x: Math.random() * 1920,
                  y: Math.random() * 1080,
                }}
                animate={{
                  y: [null, Math.random() * 1080],
                  opacity: [0, 1, 0],
                }}
                transition={{
                  duration: Math.random() * 10 + 10,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 py-16">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h1 className="text-6xl md:text-8xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
            AI Models
          </h1>
          <p className="text-xl md:text-2xl text-gray-400 max-w-2xl mx-auto">
            Discover our collection of cutting-edge AI models
          </p>
        </motion.div>

        {/* Loading State */}
        {loading && (
          <div className="text-center text-gray-400">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400"></div>
            <p className="mt-4">Loading models...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="text-center text-red-400 bg-red-900/20 border border-red-500/50 rounded-lg p-6 max-w-2xl mx-auto">
            <p className="font-semibold mb-2">Error</p>
            <p>{error}</p>
            <button
              onClick={fetchModels}
              className="mt-4 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {/* Models by Version - Grid Layout */}
        {!loading && !error && (
          <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
          {Object.entries(modelsByVersion).map(([versionKey, models]) => {
            if (models.length === 0) return null;
            
            return (
              <motion.div
                key={versionKey}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
              >
                {/* Version Header */}
                <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3 border-b border-gray-700 pb-3">
                  <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                    {versionKey.toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-500 font-normal">
                    ({models.length})
                  </span>
                </h2>

                {/* Models List */}
                <div className="space-y-2">
                  {models.map((model, index) => {
                    const IconComponent = model.icon;
                    return (
                      <motion.div
                        key={model.version}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="group bg-gray-900/30 rounded-lg p-3 hover:bg-gray-900/50 transition-all duration-200"
                      >
                        <div className="flex items-start gap-3">
                          {/* Icon */}
                          <div className={`flex-shrink-0 p-1.5 rounded-md bg-gradient-to-r ${model.color}`}>
                            <IconComponent className="w-4 h-4 text-white" />
                          </div>

                          {/* Model Info */}
                          <div className="flex-grow min-w-0">
                            <h3 className="text-sm font-semibold text-white mb-0.5">
                              {model.name}
                            </h3>
                            <p className="text-xs text-gray-400 mb-1.5">
                              {model.description}
                            </p>
                            <div className="flex items-center gap-2">
                              <code className="text-[10px] text-gray-500 bg-gray-800/50 px-1.5 py-0.5 rounded font-mono">
                                {model.version}
                              </code>
                              <button
                                onClick={() => copyToClipboard(model.version)}
                                className="p-1 rounded hover:bg-gray-700/50 transition-colors"
                                title="Copy version"
                              >
                                {copiedVersion === model.version ? (
                                  <Check className="w-3 h-3 text-green-400" />
                                ) : (
                                  <Copy className="w-3 h-3 text-gray-400 hover:text-white" />
                                )}
                              </button>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </motion.div>
            );
          })}
          </div>
        )}

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.6 }}
          className="mt-16 text-center"
        >
          <p className="text-gray-500">
            Powered by cutting-edge AI technology
          </p>
        </motion.div>
      </div>
    </div>
  );
}
