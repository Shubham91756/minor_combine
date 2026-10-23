
import React from 'react';

export default function Card({ children, className = '', hover = true, gradient = false }) {
  const baseClasses = "glass-card p-6 rounded-2xl border border-slate-200/60 dark:border-slate-700/50 shadow-lg transition-all duration-300";
  const hoverClasses = hover ? "hover:shadow-2xl hover:-translate-y-1" : "";
  const gradientClasses = gradient ? "bg-gradient-to-br from-white/80 via-indigo-50/40 to-purple-50/40 dark:from-slate-900/70 dark:via-slate-900/60 dark:to-slate-900/40" : "";

  return (
    <div className={`${baseClasses} ${hoverClasses} ${gradientClasses} ${className}`}>
      {children}
    </div>
  );
}
