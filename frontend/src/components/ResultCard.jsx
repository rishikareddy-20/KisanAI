import { useLayoutEffect, useRef, useState } from "react";
import { t } from "../i18n.js";
import { resolveAudioUrl } from "../api.js";

const LANG_TO_BCP47 = { en: "en-IN", hi: "hi-IN"};
const SEVERITY_CLASS = { high: "sev-high", medium: "sev-medium", low: "sev-low" };

export default function ResultCard({ lang, result, onReset }) {
  const { classification, plan, audio_url } = result;
  const [speaking, setSpeaking] = useState(false);
  const audioRef = useRef(null);

  // Stop playback when the result/language changes or the card unmounts, else
  // the voice keeps playing. useLayoutEffect so the button resets to "Listen"
  // before paint instead of flashing the stale "Stop" state.
  useLayoutEffect(() => {
    setSpeaking(false);
    const el = audioRef.current;
    return () => {
      if (el) {
        el.pause();
        el.currentTime = 0;
      }
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    };
  }, [result, lang]);

  const confidencePct = Math.round(classification.confidence * 100);

  const buildSpeechText = () => {
    const parts = [plan.title, plan.summary];
    if (plan.uncertain_note) parts.push(plan.uncertain_note);
    plan.sections.forEach((s) => {
      parts.push(s.label);
      parts.push(...s.items);
    });
    if (plan.safety_note) parts.push(plan.safety_note);
    return parts.filter(Boolean).join(". ");
  };

  const speak = () => {
    // Prefer backend audio, else fall back to the browser speech engine.
    if (audio_url) {
      const el = audioRef.current;
      if (el) {
        el.play();
        setSpeaking(true);
        el.onended = () => setSpeaking(false);
      }
      return;
    }
    if ("speechSynthesis" in window) {
      const utter = new SpeechSynthesisUtterance(buildSpeechText());
      utter.lang = LANG_TO_BCP47[lang] || "en-IN";
      utter.onend = () => setSpeaking(false);
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
      setSpeaking(true);
    }
  };

  const stop = () => {
    if (audioRef.current) audioRef.current.pause();
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    setSpeaking(false);
  };

  return (
    <div className="card result-card">
      <div className="result-header">
        <h2>{plan.title}</h2>
        <div className="confidence-row">
          <span className={`badge ${classification.uncertain ? "badge-warn" : "badge-ok"}`}>
            {t(lang, "confidence")}: {confidencePct}%
          </span>
        </div>
        <div className="confidence-bar">
          <div
            className={`confidence-fill ${classification.uncertain ? "warn" : ""}`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      {plan.uncertain_note && <div className="note note-warn">⚠️ {plan.uncertain_note}</div>}

      {plan.summary && <p className="summary">{plan.summary}</p>}

      <div className="voice-actions">
        {speaking ? (
          <button type="button" className="btn secondary" onClick={stop}>
            {t(lang, "stop")}
          </button>
        ) : (
          <button type="button" className="btn secondary" onClick={speak}>
            {t(lang, "listen")}
          </button>
        )}
      </div>

      {audio_url && <audio ref={audioRef} src={resolveAudioUrl(audio_url)} preload="none" />}

      {plan.sections.map((section, i) => (
        <section className="plan-section" key={i}>
          <h3>{section.label}</h3>
          <ul>
            {section.items.map((item, j) => (
              <li key={j}>{item}</li>
            ))}
          </ul>
        </section>
      ))}

      {plan.safety_note && <div className="note note-safety">🛡️ {plan.safety_note}</div>}

      <button type="button" className="btn primary full" onClick={onReset}>
        {t(lang, "tryAnother")}
      </button>
    </div>
  );
}
