// UI labels only. The treatment-plan content comes localized from the backend.

export const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "hi", label: "हिन्दी" },
];

export const UI = {
  en: {
    tagline: "Snap a leaf. Get a treatment plan.",
    chooseLanguage: "Language",
    uploadPrompt: "Upload or take a photo of the affected leaf",
    chooseImage: "Choose image",
    takePhoto: "Take photo",
    cropLabel: "Crop (optional)",
    cropPlaceholder: "e.g. Tomato",
    diagnose: "Diagnose",
    analyzing: "Analyzing leaf…",
    confidence: "Confidence",
    listen: "🔊 Listen",
    stop: "⏹ Stop",
    tryAnother: "Diagnose another leaf",
    demoModeTitle: "Demo mode",
    demoModeBody:
      "No trained model loaded yet. Predictions are simulated so you can try the full flow. Train and export a model to get real diagnoses.",
    errorGeneric: "Something went wrong. Is the backend running?",
    poweredBy: "Advisory grounded in a curated agronomy knowledge base",
  },
  hi: {
    tagline: "पत्ती की फोटो लें। उपचार योजना पाएं।",
    chooseLanguage: "भाषा",
    uploadPrompt: "प्रभावित पत्ती की फोटो अपलोड करें या खींचें",
    chooseImage: "फोटो चुनें",
    takePhoto: "फोटो खींचें",
    cropLabel: "फसल (वैकल्पिक)",
    cropPlaceholder: "जैसे टमाटर",
    diagnose: "निदान करें",
    analyzing: "पत्ती की जाँच हो रही है…",
    confidence: "विश्वास",
    listen: "🔊 सुनें",
    stop: "⏹ रोकें",
    tryAnother: "दूसरी पत्ती जाँचें",
    demoModeTitle: "डेमो मोड",
    demoModeBody:
      "अभी कोई प्रशिक्षित मॉडल लोड नहीं है। पूर्वानुमान नकली हैं ताकि आप पूरा प्रवाह आज़मा सकें। असली निदान के लिए मॉडल प्रशिक्षित और निर्यात करें।",
    errorGeneric: "कुछ गड़बड़ हो गई। क्या बैकएंड चल रहा है?",
    poweredBy: "सलाह एक चयनित कृषि ज्ञान-आधार पर आधारित है",
  },
};

export function t(lang, key) {
  return (UI[lang] || UI.en)[key] ?? UI.en[key];
}