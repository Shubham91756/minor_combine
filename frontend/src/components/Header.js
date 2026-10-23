
import React from 'react';

export default function Header({ theme, onToggleTheme, demoMode, onToggleDemo, systemStatus }) {
  const statusChip = (label, active) => (
    <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/70 dark:bg-slate-900/70 border border-white/40 dark:border-slate-700/50 text-xs">
      <span className={`h-2 w-2 rounded-full ${active === null ? 'bg-slate-300' : active ? 'bg-emerald-500' : 'bg-rose-500'} animate-pulse`}></span>
      <span className="text-slate-700 dark:text-slate-200">{label}</span>
    </div>
  );

  return (
    <header className="relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-500"></div>
      <div className="absolute inset-0 opacity-30">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>

      <div className="relative max-w-7xl mx-auto px-6 py-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-3">
              <div className="h-10 w-10 rounded-2xl bg-white/20 backdrop-blur-md flex items-center justify-center text-white shadow-lg">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
              </div>
              <div>
                <h1 className="text-3xl font-semibold text-white font-display">Parkinson’s Models Suite</h1>
                <p className="text-sm text-indigo-100">Advanced AI · Research Dashboard · Multi-Modal Screening</p>
              </div>
            </div>

            <nav className="mt-4 flex flex-wrap gap-4 text-sm text-indigo-100">
              <a className="hover:text-white transition" href="#handwriting">Handwriting</a>
              <a className="hover:text-white transition" href="#motor">Motor</a>
              <a className="hover:text-white transition" href="#gait">Gait</a>
            </nav>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              className={`px-4 py-2 rounded-full text-xs font-semibold border border-white/40 ${demoMode ? 'bg-white text-indigo-700' : 'bg-white/20 text-white'} transition`}
              onClick={onToggleDemo}
            >
              {demoMode ? 'Demo Mode: On' : 'Demo Mode: Off'}
            </button>
            <button
              className="h-10 w-10 rounded-full bg-white/20 border border-white/40 text-white flex items-center justify-center hover:bg-white/30 transition"
              onClick={onToggleTheme}
              title="Toggle light/dark mode"
            >
              {theme === 'dark' ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m8.66-10H21M3 12H2m15.364-6.364l.707.707M6.343 17.657l-.707.707m12.728 0l-.707-.707M6.343 6.343l-.707-.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12.79A9 9 0 1111.21 3a7 7 0 109.79 9.79z" />
                </svg>
              )}
            </button>
          </div>
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          {statusChip('Handwriting model', systemStatus && systemStatus.handwriting)}
          {statusChip('Motor model', systemStatus && systemStatus.motor)}
          {statusChip('Gait model', systemStatus && systemStatus.gait)}
        </div>
      </div>
    </header>
  );
}
