import { useRef } from "react";
import { t } from "../i18n.js";

export default function UploadCard({ lang, preview, onSelectFile, crop, onCropChange }) {
  const fileInput = useRef(null);
  const cameraInput = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (file) onSelectFile(file);
  };

  return (
    <div className="card upload-card">
      <p className="upload-prompt">{t(lang, "uploadPrompt")}</p>

      {preview ? (
        <img className="preview" src={preview} alt="Selected leaf" />
      ) : (
        <div className="preview placeholder" aria-hidden="true">🌿</div>
      )}

      <div className="upload-actions">
        <button type="button" className="btn secondary" onClick={() => fileInput.current?.click()}>
          🖼️ {t(lang, "chooseImage")}
        </button>
        <button type="button" className="btn secondary" onClick={() => cameraInput.current?.click()}>
          📷 {t(lang, "takePhoto")}
        </button>
      </div>

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        hidden
        onChange={handleFile}
      />
      <input
        ref={cameraInput}
        type="file"
        accept="image/*"
        capture="environment"
        hidden
        onChange={handleFile}
      />

      <label className="crop-field">
        <span>{t(lang, "cropLabel")}</span>
        <input
          type="text"
          value={crop}
          placeholder={t(lang, "cropPlaceholder")}
          onChange={(e) => onCropChange(e.target.value)}
        />
      </label>
    </div>
  );
}
