module.exports = {
  darkMode: 'class',
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    './public/index.html'
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['\"Source Sans 3\"', 'system-ui', 'sans-serif'],
        display: ['\"Sora\"', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        glow: '0 0 40px rgba(99, 102, 241, 0.25)',
      },
    },
  },
  plugins: [],
}
