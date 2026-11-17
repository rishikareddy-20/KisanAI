import { useEffect, useState } from "react";
import LanguageSelector from "./components/LanguageSelector.jsx";
import UploadCard from "./components/UploadCard.jsx";
import ResultCard from "./components/ResultCard.jsx";
import { diagnose, getHealth } from "./api.js";
import { t } from "./i18n.js";

export default function App() {
  const [lang, setLang] = useState("en");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [crop, setCrop] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const onSelectFile = (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  };

  const runDiagnose = async (language) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await diagnose({ file, language, crop });
      setResult(res);
    } catch (e) {
      setError(e.message || t(language, "errorGeneric"));
    } finally {
      setLoading(false);
    }
  };

  const onDiagnose = () => runDiagnose(lang);

  // Re-fetch on language switch so the plan content is regenerated in the new
  // language; without this only the static UI labels would change.
  useEffect(() => {
    if (result && file) {
      runDiagnose(lang);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang]);

  const onReset = () => {
    setResult(null);
    setError(null);
    setFile(null);
    setPreview(null);
  };

  const demoMode = health?.vision_is_mock;

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="logo">🌱</span>
          <div>
            <h1>KisanAI</h1>
            <p className="tagline">{t(lang, "tagline")}</p>
          </div>
        </div>
        <LanguageSelector value={lang} onChange={setLang} />
      </header>

      {demoMode && (
        <div className="note note-info">
          <strong>{t(lang, "demoModeTitle")}:</strong> {t(lang, "demoModeBody")}
        </div>
      )}

      <main className="app-main">
        {!result && (
          <>
            <UploadCard
              lang={lang}
              preview={preview}
              onSelectFile={onSelectFile}
              crop={crop}
              onCropChange={setCrop}
            />
            <button
              type="button"
              className="btn primary full"
              disabled={!file || loading}
              onClick={onDiagnose}
            >
              {loading ? t(lang, "analyzing") : t(lang, "diagnose")}
            </button>
            {error && <div className="note note-warn">{error}</div>}
          </>
        )}

        {result && <ResultCard lang={lang} result={result} onReset={onReset} />}
      </main>

      <footer className="app-footer">
        <span>{t(lang, "poweredBy")}</span>
        {health && (
          <span className="health-line">
            Advisory: {health.advisory_provider} · {health.num_classes} classes
          </span>
        )}
      </footer>
    </div>
  );
}
