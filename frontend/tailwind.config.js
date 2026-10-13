/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ink: {
          teal: '#0E3B43',
          dark: '#082328',
          light: '#165763',
        },
        signal: {
          teal: '#1F8A9A',
          hover: '#167382',
        },
        paper: {
          DEFAULT: '#F3F7F8',
          subtle: '#E8EFF1',
          card: '#FFFFFF',
          dark: '#0F172A',
          cardDark: '#1E293B',
        },
        urgent: {
          coral: '#C0392B',
          light: '#FDEDEC',
        },
        caution: {
          amber: '#C98A00',
          light: '#FEF9E7',
        },
        ok: {
          sage: '#2F7D5B',
          light: '#EAF5EF',
        },
      },
      fontFamily: {
        sans: ['"Atkinson Hyperlegible"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"SF Mono"', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
