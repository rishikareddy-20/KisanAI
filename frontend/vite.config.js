import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api → FastAPI backend so the frontend can call it
// without CORS friction during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
