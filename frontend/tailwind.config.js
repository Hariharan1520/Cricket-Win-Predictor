/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: '#090d16',
        panelBg: '#111827',
        cardBg: '#1f2937',
        borderCol: '#374151',
      },
    },
  },
  plugins: [],
}
