/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        primary: '#00D4FF',
        accent: '#00FFFF',
        alert: '#FF4B2B',
        dark: '#00050A',
        'dark-card': '#0a1628',
        'dark-border': '#0d2847',
        surface: '#061020',
        'hud-dim': '#4a7c9b',
        'hud-text': '#b0d4e8',
      },
      fontFamily: {
        sans: ['Pretendard', 'system-ui', '-apple-system', 'sans-serif'],
        prose: ['Pretendard', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        hud: ['Orbitron', 'JetBrains Mono', 'monospace'],
        pixel: ['Mona12 Text KR', 'Mona12', 'monospace'],
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'fade-in-up': 'fadeInUp 0.8s ease-out forwards',
        'fade-in': 'fadeIn 1s ease-out forwards',
        'typing': 'typing 2s steps(30) forwards',
        'blink': 'blink 1s step-end infinite',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'scanline': 'scanline 8s linear infinite',
        'glow-pulse': 'glowPulse 3s ease-in-out infinite',
        'hud-border': 'hudBorder 2s ease-out forwards',
        'flicker': 'flicker 4s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(30px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        typing: {
          '0%': { width: '0' },
          '100%': { width: '100%' },
        },
        blink: {
          '50%': { borderColor: 'transparent' },
        },
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        glowPulse: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '1' },
        },
        hudBorder: {
          '0%': { 'clip-path': 'inset(0 100% 100% 0)' },
          '50%': { 'clip-path': 'inset(0 0 100% 0)' },
          '100%': { 'clip-path': 'inset(0 0 0 0)' },
        },
        flicker: {
          '0%, 100%': { opacity: '1' },
          '92%': { opacity: '1' },
          '93%': { opacity: '0.8' },
          '94%': { opacity: '1' },
          '96%': { opacity: '0.9' },
          '97%': { opacity: '1' },
        },
      },
      boxShadow: {
        'hud': '0 0 15px rgba(0, 212, 255, 0.15), inset 0 0 15px rgba(0, 212, 255, 0.05)',
        'hud-hover': '0 0 25px rgba(0, 212, 255, 0.25), inset 0 0 25px rgba(0, 212, 255, 0.08)',
        'glow': '0 0 10px rgba(0, 212, 255, 0.3)',
        'glow-strong': '0 0 20px rgba(0, 212, 255, 0.5)',
      },
    },
  },
  plugins: [],
};
