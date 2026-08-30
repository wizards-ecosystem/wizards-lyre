import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "node_modules"] },
  {
    files: ["src/**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],

      // The two rules below ship as errors in the React Compiler-era plugin.
      // This app does not use the compiler, and every current report is a
      // false positive against working, tested code, so they are advisory
      // here rather than CI-blocking:
      //
      // - set-state-in-effect fires on the deliberate "reset derived UI state
      //   when the selected take or project changes" effects. That is the
      //   intended behavior, not a cascading-render bug.
      // - immutability fires on `audioEl.currentTime = next` (assigning to a
      //   DOM element property is correct) and on a hoisted function
      //   declaration referenced from a closure that only runs later.
      //
      // They are kept on as warnings because a genuinely new violation is
      // still worth seeing.
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/immutability": "warn",
    },
  },
  {
    // Tests reach into mock internals and assert on loosely-typed fixtures.
    files: ["src/__tests__/**/*.{ts,tsx}", "src/test/**/*.{ts,tsx}"],
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
);
