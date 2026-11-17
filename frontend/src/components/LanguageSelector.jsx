import { LANGUAGES } from "../i18n.js";

export default function LanguageSelector({ value, onChange }) {
  return (
    <div className="lang-selector" role="group" aria-label="Select language">
      {LANGUAGES.map((lang) => (
        <button
          key={lang.code}
          type="button"
          className={`lang-pill ${value === lang.code ? "active" : ""}`}
          onClick={() => onChange(lang.code)}
        >
          {lang.label}
        </button>
      ))}
    </div>
  );
}
