/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      borderRadius: {
        lg: "0.75rem",
      },
      colors: {
        bg: "#0b0d10",
        panel: "#11151a",
        panel2: "#151b22",
        text: "#e7edf5",
        muted: "#a7b4c4",
        brand: "#77b7ff",
        danger: "#ff5a6a",
      },
    },
  },
  plugins: [],
};

