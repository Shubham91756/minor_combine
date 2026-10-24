
import React, { useEffect, useMemo, useState } from 'react';
import './App.css';
import Header from './components/Header';
import Card from './components/Card';

const DEMO_MOTOR_JSON = `{
  "features": {
    "c1_mean": 0.012,
    "c1_std": 0.072,
    "c1_median": 0.001,
    "c1_min": -0.223,
    "c1_max": 0.198,
    "c6_energy": 620.5,
    "c6_std": 0.295,
    "acc_mag_mean": 0.08,
    "acc_mag_std": 0.09,
    "nrows": 1200,
    "duration": 11.9,
    "median_dt": 0.01
  }
}`;

const MAX_FILE_MB = 10;
const API_KEY = process.env.REACT_APP_API_KEY || '';
const PERSIST_HISTORY = process.env.REACT_APP_PERSIST_HISTORY === '1';

const apiHeaders = () => (API_KEY ? { 'X-API-Key': API_KEY } : {});

function App() {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('pd_theme');
    if (saved) return saved;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  });

  const [demoMode, setDemoMode] = useState(false);

  const xaiEnabled = (process.env.REACT_APP_XAI || '1') !== '0';

  const [hwFile, setHwFile] = useState(null);
  const [hwPreview, setHwPreview] = useState(null);
  const [hwMeta, setHwMeta] = useState(null);
  const [hwHeatmapOpacity, setHwHeatmapOpacity] = useState(0.55);

  const [motorJson, setMotorJson] = useState('{}');
  const [motorJsonFile, setMotorJsonFile] = useState(null);
  const [motorCsvFile, setMotorCsvFile] = useState(null);
  const [motorCsvPreview, setMotorCsvPreview] = useState(null);

  const [motorTsFiles, setMotorTsFiles] = useState([]);
  const [motorTsPreview, setMotorTsPreview] = useState(null);

  const [gaitFile, setGaitFile] = useState(null);
  const [gaitPreview, setGaitPreview] = useState(null);

  const [loading, setLoading] = useState({ active: false, modality: null, stage: null });
  const [result, setResult] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  const [fileErrors, setFileErrors] = useState({ handwriting: null, motorJson: null, motorCsv: null, gait: null });
  const [history, setHistory] = useState(() => {
    if (!PERSIST_HISTORY) return [];
    try {
      const saved = localStorage.getItem('pd_history');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  const [modalityResults, setModalityResults] = useState({
    handwriting: null,
    motor: null,
    gait: null,
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('pd_theme', theme);
  }, [theme]);

  useEffect(() => {
    if (PERSIST_HISTORY) {
      localStorage.setItem('pd_history', JSON.stringify(history.slice(0, 25)));
    }
  }, [history]);

  useEffect(() => {
    async function loadInfo() {
      const backend = process.env.REACT_APP_API_URL || '';
      try {
        const res = await fetch(`${backend}/info`, { headers: apiHeaders() });
        if (res.ok) {
          const data = await res.json();
          setModelInfo(data);
        }
      } catch (e) {
        // ignore
      }
    }
    loadInfo();
  }, []);

  useEffect(() => {
    if (demoMode && motorJson.trim() === '{}' ) {
      setMotorJson(DEMO_MOTOR_JSON);
    }
  }, [demoMode, motorJson]);

  const formatModelName = (path) => {
    if (!path) return 'n/a';
    const parts = String(path).split(/[/\\]/);
    return parts[parts.length - 1];
  };

  const formatPct = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) return 'n/a';
    return `${(value * 100).toFixed(1)}%`;
  };

  const getRiskBand = (score) => {
    if (score === null || score === undefined || Number.isNaN(score)) {
      return { label: 'n/a', tone: 'neutral', className: 'text-slate-500' };
    }
    if (score >= 0.65) return { label: 'High', tone: 'high', className: 'text-rose-600' };
    if (score >= 0.45) return { label: 'Medium', tone: 'medium', className: 'text-amber-600' };
    return { label: 'Low', tone: 'low', className: 'text-emerald-600' };
  };

  const appendHistory = (entry) => {
    setHistory((prev) => [{ id: Date.now(), ...entry }, ...prev].slice(0, 25));
  };

  const motorJsonValid = useMemo(() => {
    try {
      const parsed = JSON.parse(motorJson);
      if (!parsed || typeof parsed !== 'object') return false;
      const obj = (parsed.features && typeof parsed.features === 'object') ? parsed.features : parsed;
      return obj && typeof obj === 'object' && Object.keys(obj).length > 0;
    } catch (e) {
      return false;
    }
  }, [motorJson]);

  const handwritingMeta = modelInfo && modelInfo.handwriting_model ? modelInfo.handwriting_model : null;
  const handwritingModelNames = handwritingMeta && Array.isArray(handwritingMeta.paths) && handwritingMeta.paths.length
    ? handwritingMeta.paths.map(formatModelName).join(' + ')
    : 'ensemble';

  const postFile = async (url, file) => {
    const form = new FormData();
    form.append('file', file);

    const normalizeBase = (base) => {
      if (!base) return '';
      const trimmed = String(base).trim();
      if (!trimmed) return '';
      if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
      return `http://${trimmed.replace(/\/+$/, '')}`;
    };
    const backendEnv = normalizeBase(process.env.REACT_APP_API_URL || '');

    const tryFetch = async (base) => {
      const fullUrl = base ? `${base}${url}` : url;
      const res = await fetch(fullUrl, { method: 'POST', body: form, headers: apiHeaders() });
      const text = await res.text();
      try {
        const parsed = JSON.parse(text);
        if (!res.ok) {
          throw new Error(parsed && parsed.detail ? parsed.detail : `Request failed with status ${res.status}`);
        }
        return parsed;
      } catch (err) {
        if (!res.ok) {
          throw new Error(`Request failed with status ${res.status}: ${text.substring(0, 800)}`);
        }
        throw err;
      }
    };

    try {
      return await tryFetch(backendEnv);
    } catch (err) {
      // Fallbacks for local dev when proxy/env is misconfigured.
      // Try 8001 first (matches `frontend/package.json` proxy), then 8000.
      const fallbacks = ['http://127.0.0.1:8001', 'http://127.0.0.1:8000'];
      for (const fallback of fallbacks) {
        if (backendEnv && backendEnv === fallback) continue;
        try {
          return await tryFetch(fallback);
        } catch (e) {
          // keep trying
        }
      }
      throw err;
    }
  };

  const postJSON = async (url, payload) => {
    const normalizeBase = (base) => {
      if (!base) return '';
      const trimmed = String(base).trim();
      if (!trimmed) return '';
      if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
      return `http://${trimmed.replace(/\/+$/, '')}`;
    };
    const backendEnv = normalizeBase(process.env.REACT_APP_API_URL || '');

    const tryFetch = async (base) => {
      const fullUrl = base ? `${base}${url}` : url;
      const res = await fetch(fullUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...apiHeaders() },
        body: JSON.stringify(payload),
      });
      const text = await res.text();
      let parsed = null;
      try {
        parsed = JSON.parse(text);
      } catch (e) {
        parsed = null;
      }
      if (!res.ok) {
        throw new Error((parsed && parsed.detail) ? parsed.detail : `Request failed with status ${res.status}`);
      }
      return parsed ?? {};
    };

    try {
      return await tryFetch(backendEnv);
    } catch (err) {
      const fallbacks = ['http://127.0.0.1:8001', 'http://127.0.0.1:8000'];
      for (const fallback of fallbacks) {
        if (backendEnv && backendEnv === fallback) continue;
        try {
          return await tryFetch(fallback);
        } catch (e) {
          // continue
        }
      }
      throw err;
    }
  };

  const handleHandwritingChange = (e) => {
    const file = e.target.files && e.target.files[0];
    setHwFile(file || null);
    setHwPreview(null);
    setHwMeta(null);
    setFileErrors((prev) => ({ ...prev, handwriting: null }));

    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setFileErrors((prev) => ({ ...prev, handwriting: 'Only image files are supported.' }));
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFileErrors((prev) => ({ ...prev, handwriting: `Max file size is ${MAX_FILE_MB}MB.` }));
      return;
    }

    const reader = new FileReader();
    reader.onload = () => setHwPreview(reader.result);
    reader.readAsDataURL(file);

    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      setHwMeta({ width: img.width, height: img.height });
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  const handleMotorCsvChange = (e) => {
    const file = e.target.files && e.target.files[0];
    setMotorCsvFile(file || null);
    setMotorCsvPreview(null);
    setFileErrors((prev) => ({ ...prev, motorCsv: null }));

    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setFileErrors((prev) => ({ ...prev, motorCsv: 'CSV file required.' }));
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFileErrors((prev) => ({ ...prev, motorCsv: `Max file size is ${MAX_FILE_MB}MB.` }));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      const preview = text.split(/\r?\n/).slice(0, 4).join('\n');
      setMotorCsvPreview(preview);
    };
    reader.readAsText(file);
  };

  const handleMotorJsonFileChange = (e) => {
    const file = e.target.files && e.target.files[0];
    setMotorJsonFile(file || null);
    setFileErrors((prev) => ({ ...prev, motorJson: null }));

    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.json')) {
      setFileErrors((prev) => ({ ...prev, motorJson: 'JSON file required.' }));
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFileErrors((prev) => ({ ...prev, motorJson: `Max file size is ${MAX_FILE_MB}MB.` }));
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = String(reader.result || '');
        const parsed = JSON.parse(text);
        const obj = (parsed && typeof parsed === 'object' && parsed.features && typeof parsed.features === 'object')
          ? parsed.features
          : parsed;
        if (!obj || typeof obj !== 'object' || !Object.keys(obj).length) {
          setFileErrors((prev) => ({ ...prev, motorJson: 'JSON does not contain usable motor features.' }));
          return;
        }
        setMotorJson(JSON.stringify(parsed, null, 2));
      } catch (err) {
        setFileErrors((prev) => ({ ...prev, motorJson: 'Invalid JSON file.' }));
      }
    };
    reader.readAsText(file);
  };

  const loadDemoMotorCsv = () => {
    try {
      let features = null;
      if (motorJsonValid) {
        const parsed = JSON.parse(motorJson);
        features = (parsed && parsed.features && typeof parsed.features === 'object')
          ? parsed.features
          : parsed;
      }
      if (!features || typeof features !== 'object' || !Object.keys(features).length) {
        features = JSON.parse(DEMO_MOTOR_JSON).features;
      }
      const keys = Object.keys(features);
      const row = keys.map((k) => features[k]);
      const csv = `${keys.join(',')}\n${row.join(',')}\n`;
      const file = new File([csv], 'motor_demo_features.csv', { type: 'text/csv' });
      setMotorCsvFile(file);
      setMotorCsvPreview(csv.split(/\r?\n/).slice(0, 4).join('\n'));
      setFileErrors((prev) => ({ ...prev, motorCsv: null }));
    } catch (err) {
      setFileErrors((prev) => ({ ...prev, motorCsv: 'Unable to generate demo CSV. Use file upload.' }));
    }
  };

  
  const handleMotorTsChange = (e) => {
    const file = e.target.files && e.target.files[0];
    const files = Array.from(e.target.files || []);
    setMotorTsFiles(files);
    setMotorTsPreview(null);
    setFileErrors((prev) => ({ ...prev, motorTs: null }));

    if (!file) return;
    if (!files[0].name.toLowerCase().endsWith('.txt')) {
      setFileErrors((prev) => ({ ...prev, motorTs: 'Text (.txt) timeseries file required.' }));
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFileErrors((prev) => ({ ...prev, motorTs: `Max file size is ${MAX_FILE_MB}MB.` }));
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      const preview = text.split(/\r?\n/).slice(0, 5).join('\n');
      setMotorTsPreview(preview);
    };
    reader.readAsText(file);

  };

  const loadDemoMotorTimeseries = () => {
    const lines = ['0.00,0.02,0.01,0.03,0.01,0.00,0.02'];
    for (let i = 1; i <= 300; i += 1) {
      const t = (i * 0.01).toFixed(2);
      const s1 = (0.03 * Math.sin(i / 12)).toFixed(4);
      const s2 = (0.02 * Math.cos(i / 15)).toFixed(4);
      const s3 = (0.04 * Math.sin(i / 9)).toFixed(4);
      const s4 = (0.01 * Math.cos(i / 7)).toFixed(4);
      const s5 = (0.02 * Math.sin(i / 11)).toFixed(4);
      const s6 = (0.01 * Math.cos(i / 6)).toFixed(4);
      lines.push(`${t},${s1},${s2},${s3},${s4},${s5},${s6}`);
    }
    const txt = `${lines.join('\n')}\n`;
    const file = new File([txt], 'motor_demo_timeseries.txt', { type: 'text/plain' });
    setMotorTsFiles([file]);
    setMotorTsPreview(txt.split(/\r?\n/).slice(0, 5).join('\n'));
    setFileErrors((prev) => ({ ...prev, motorTs: null }));
  };

  const handleGaitChange = (e) => {
    const file = e.target.files && e.target.files[0];
    setGaitFile(file || null);
    setGaitPreview(null);
    setFileErrors((prev) => ({ ...prev, gait: null }));

    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.txt')) {
      setFileErrors((prev) => ({ ...prev, gait: 'Text (.txt) file required.' }));
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFileErrors((prev) => ({ ...prev, gait: `Max file size is ${MAX_FILE_MB}MB.` }));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      const preview = text.split(/\r?\n/).slice(0, 5).join('\n');
      setGaitPreview(preview);
    };
    reader.readAsText(file);
  };

  const handleHandwriting = async (e) => {
    e.preventDefault();
    if (!hwFile) return alert('Select an image first');
    setLoading({ active: true, modality: 'handwriting', stage: 'Processing image' });
    setResult(null);
    try {
      const hwUrl = xaiEnabled ? '/predict/handwriting?explain=1' : '/predict/handwriting';
      const data = await postFile(hwUrl, hwFile);
      setResult({ endpoint: 'handwriting', data });
      setModalityResults((prev) => ({ ...prev, handwriting: data }));
      appendHistory({ modality: 'Handwriting', label: data.label, prob: data.prob_pos, ts: new Date().toISOString() });
    } catch (err) {
      setResult({ error: String(err) });
    } finally {
      setLoading({ active: false, modality: null, stage: null });
    }
  };

  const handleMotorJson = async (e) => {
    e.preventDefault();
    setLoading({ active: true, modality: 'motor', stage: 'Parsing JSON' });
    setResult(null);
    try {
      let parsed = JSON.parse(motorJson);
      let featuresObj = null;
      if (parsed && typeof parsed === 'object') {
        if (parsed.features && typeof parsed.features === 'object') {
          featuresObj = parsed.features;
        } else {
          featuresObj = parsed;
        }
      } else {
        throw new Error('Invalid JSON payload for motor features');
      }
      if (!featuresObj || Object.keys(featuresObj).length === 0) {
        throw new Error('Motor features are empty. Click \"Load demo JSON\" or provide feature values.');
      }
      const payload = { features: featuresObj };
      const data = await postJSON('/predict/motor', payload);
      setResult({ endpoint: 'motor_json', data });
      if (data && data.prob_pos != null) {
        setModalityResults((prev) => ({ ...prev, motor: data }));
        appendHistory({ modality: 'Motor', label: data.label, prob: data.prob_pos, ts: new Date().toISOString() });
      }
    } catch (err) {
      setResult({ error: String(err) });
    } finally {
      setLoading({ active: false, modality: null, stage: null });
    }
  };

  const handleMotorCsv = async (e) => {
    e.preventDefault();
    if (!motorCsvFile) return alert('Select a CSV file first');
    setLoading({ active: true, modality: 'motor', stage: 'Processing CSV' });
    setResult(null);
    try {
      const data = await postFile('/predict/motor_csv', motorCsvFile);
      setResult({ endpoint: 'motor_csv', data });
      if (data && Array.isArray(data.predictions) && data.predictions.length > 0) {
        const avg = data.predictions.reduce((acc, row) => acc + (row.prob_pos || 0), 0) / data.predictions.length;
        appendHistory({ modality: 'Motor CSV', label: 'batch', prob: avg, ts: new Date().toISOString() });
      }
    } catch (err) {
      setResult({ error: String(err) });
    } finally {
      setLoading({ active: false, modality: null, stage: null });
    }
  
  };

  const handleMotorTimeseries = async (e) => {
    e.preventDefault();
    if (!motorTsFiles.length) return alert('Select one or more motor timeseries (.txt) file(s) first');
    setLoading({ active: true, modality: 'motor', stage: 'Analyzing timeseries' });
    setResult(null);
    try {
      let dataRaw;
      if (motorTsFiles.length === 1) {
        dataRaw = await postFile('/predict/motor_timeseries', motorTsFiles[0]);
      } else {
        const resNew = await fetch('/session/motor/new', { headers: apiHeaders() });
        const newJson = await resNew.json();
        const sessionId = newJson.session_id;
        for (const f of motorTsFiles) {
          await postFile(`/session/motor/${sessionId}/upload`, f);
        }
        const resSum = await fetch(`/session/motor/${sessionId}/summary`, { headers: apiHeaders() });
        dataRaw = await resSum.json();
      }
      // For dashboard display: use binary patient probability if available
      const data = { ...dataRaw };
      if (data.prob_patient != null) {
        data.prob_pos = data.prob_patient;
      }
      if (data.binary_label) {
        data.label = data.binary_label;
      }
      setResult({ endpoint: 'motor_timeseries', data });
      if (data && data.prob_pos != null) {
        setModalityResults((prev) => ({ ...prev, motor: data }));
        appendHistory({ modality: 'Motor (timeseries)', label: data.label, prob: data.prob_pos, ts: new Date().toISOString() });
      }
    } catch (err) {
      setResult({ error: String(err) });
    } finally {
      setLoading({ active: false, modality: null, stage: null });
    }
  };

  const handleGait = async (e) => {
    e.preventDefault();
    if (!gaitFile) return alert('Select a gait timeseries file first');
    setLoading({ active: true, modality: 'gait', stage: 'Analyzing signal' });
    setResult(null);
    try {
      const gaitUrl = xaiEnabled ? '/predict/gait?explain=1' : '/predict/gait';
      const data = await postFile(gaitUrl, gaitFile);
      setResult({ endpoint: 'gait', data });
      if (data && data.prob_pos != null) {
        setModalityResults((prev) => ({ ...prev, gait: data }));
        appendHistory({ modality: 'Gait', label: data.label || data.verdict, prob: data.prob_pos, ts: new Date().toISOString() });
      }
    } catch (err) {
      setResult({ error: String(err) });
    } finally {
      setLoading({ active: false, modality: null, stage: null });
    }
  };

  const fusion = useMemo(() => {
    const parts = [];
    if (modalityResults.handwriting && modalityResults.handwriting.prob_pos != null) {
      parts.push({
        key: 'handwriting',
        label: 'Handwriting',
        score: modalityResults.handwriting.prob_pos,
        threshold: modalityResults.handwriting.threshold != null ? modalityResults.handwriting.threshold : 0.5,
        weight: 0.4,
      });
    }
    if (modalityResults.motor && modalityResults.motor.prob_pos != null) {
      parts.push({
        key: 'motor',
        label: 'Motor',
        score: modalityResults.motor.prob_pos,
        threshold: 0.5,
        weight: 0.3,
      });
    }
    if (modalityResults.gait && modalityResults.gait.prob_pos != null) {
      parts.push({
        key: 'gait',
        label: 'Gait',
        score: modalityResults.gait.prob_pos,
        threshold: 0.5,
        weight: 0.3,
      });
    }
    if (!parts.length) return null;

    const weightSum = parts.reduce((sum, p) => sum + p.weight, 0);
    const score = parts.reduce((sum, p) => sum + p.score * p.weight, 0) / weightSum;
    const patientVotes = parts.filter((p) => p.score >= p.threshold).length;
    const agreement = patientVotes === 0
      ? 'All modalities lean healthy'
      : (patientVotes === parts.length
        ? 'All modalities indicate patient'
        : `${patientVotes}/${parts.length} modalities indicate patient`);
    return { score, parts, patientVotes, agreement };
  }, [modalityResults]);

  const fusionBand = fusion ? getRiskBand(fusion.score) : null;

  const fusionConfidence = fusion && fusion.score != null
    ? Math.abs(fusion.score - 0.5)
    : null;

  const confidenceLabel = fusionConfidence == null
    ? 'n/a'
    : (fusionConfidence < 0.08 ? 'Low' : (fusionConfidence < 0.18 ? 'Medium' : 'High'));

  const insightSummary = useMemo(() => {
    if (!fusion) return 'Run at least one modality to generate an AI insight summary.';
    const highSignals = fusion.parts.filter((p) => p.score >= p.threshold).map((p) => p.label);
    const lowSignals = fusion.parts.filter((p) => p.score < p.threshold).map((p) => p.label);
    if (highSignals.length === fusion.parts.length) {
      return `Consistent patient indicators across ${fusion.parts.length} modalities. Confidence is boosted by cross-modality agreement.`;
    }
    if (highSignals.length === 0) {
      return `All available modalities align toward healthy. Continue longitudinal monitoring for confirmation.`;
    }
    if (highSignals.length === 1) {
      return `Signal is driven primarily by ${highSignals[0]}. Consider a repeat test or confirm with other modalities.`;
    }
    return `Mixed signals: patient indicators in ${highSignals.join(', ')} while ${lowSignals.join(', ')} suggest healthy. Interpret with caution.`;
  }, [fusion]);
  const badgeForConfidence = (ci) => {
    const level = (ci && ci.level) ? String(ci.level) : 'n/a';
    const base = 'text-xs px-2.5 py-1 rounded-full border';
    if (level === 'High') return { label: 'High confidence', className: `${base} bg-emerald-50/70 text-emerald-700 border-emerald-200/60 dark:bg-emerald-500/10 dark:text-emerald-200 dark:border-emerald-500/30` };
    if (level === 'Medium') return { label: 'Medium confidence', className: `${base} bg-amber-50/70 text-amber-700 border-amber-200/60 dark:bg-amber-500/10 dark:text-amber-200 dark:border-amber-500/30` };
    if (level === 'Low') return { label: 'Low confidence', className: `${base} bg-rose-50/70 text-rose-700 border-rose-200/60 dark:bg-rose-500/10 dark:text-rose-200 dark:border-rose-500/30` };
    return { label: 'Confidence n/a', className: `${base} bg-slate-50/70 text-slate-600 border-slate-200/60 dark:bg-slate-800/60 dark:text-slate-300 dark:border-slate-700/50` };
  };

  function prettyFeatureName(name) {
    if (!name) return 'feature';
    return String(name)
      .replace(/^cop_/, 'COP ')
      .replace(/_/g, ' ')
      .replace(/\bspm\b/i, 'steps/min')
      .replace(/\bstd\b/i, 'SD');
  }

  const resolveAssetUrl = (relativeUrl) => {
    if (!relativeUrl) return null;
    const u = String(relativeUrl);
    if (u.startsWith('http://') || u.startsWith('https://') || u.startsWith('data:')) return u;
    const backend = process.env.REACT_APP_API_URL || '';
    return backend ? `${backend}${u}` : u;
  };

  const buildClinicalNarrative = () => {
    if (!fusion) {
      return { title: 'Awaiting data', text: 'Run at least one modality to generate a clinically transparent AI insight summary.' };
    }

    const partByKey = Object.fromEntries(fusion.parts.map((p) => [p.key, p]));
    const hw = partByKey.handwriting;
    const motor = partByKey.motor;
    const gait = partByKey.gait;

    const hwHi = hw && hw.score >= hw.threshold;
    const motorHi = motor && motor.score >= motor.threshold;
    const gaitHi = gait && gait.score >= gait.threshold;

    const clauses = [];

    if (hw) {
      clauses.push(hwHi
        ? `Observed elevated handwriting instability (patient-likeness score ${formatPct(hw.score)}).`
        : `Handwriting remains within normal variance (score ${formatPct(hw.score)}).`
      );
    }
    if (gait) {
      clauses.push(gaitHi
        ? `Gait signal indicates irregular cadence/pressure dynamics (score ${formatPct(gait.score)}).`
        : `Gait cadence/pressure dynamics remain within typical variance (score ${formatPct(gait.score)}).`
      );
    }
    if (motor) {
      clauses.push(motorHi
        ? `Upper-extremity motor features show elevated variability (score ${formatPct(motor.score)}).`
        : `Upper-extremity motor features remain within expected range (score ${formatPct(motor.score)}).`
      );
    }

    let recommendation = '';
    if (hwHi && gait && !gaitHi) {
      recommendation = 'Recommend focus on upper-extremity motor tasks and repeat handwriting capture under standardized conditions.';
    } else if (gaitHi && hw && !hwHi) {
      recommendation = 'Recommend gait re-test on a controlled walkway and review footwear/surface effects.';
    } else if (fusion.patientVotes === 0) {
      recommendation = 'Recommendation: continue longitudinal monitoring; repeat tests if symptoms change.';
    } else if (fusion.patientVotes === fusion.parts.length) {
      recommendation = 'Recommendation: consider clinical screening workflow and clinician review of raw samples.';
    } else {
      recommendation = 'Recommendation: mixed signals—repeat the discordant modality and consider adding another modality for confirmation.';
    }

    const title = `Fusion risk: ${getRiskBand(fusion.score).label} (${formatPct(fusion.score)})`;
    const text = `${clauses.join(' ')} ${recommendation}`;
    return { title, text };
  };

  const clinicalInsight = useMemo(() => buildClinicalNarrative(), [fusion, modalityResults]);

  const fusionContributions = useMemo(() => {
    if (!fusion) return [];
    return fusion.parts.map((p) => ({
      key: p.key,
      label: p.label,
      weight: p.weight,
      delta: (p.score - p.threshold),
      contribution: p.weight * (p.score - p.threshold),
    }));
  }, [fusion]);


  const systemStatus = {
    handwriting: modelInfo && modelInfo.handwriting_model ? modelInfo.handwriting_model.loaded : null,
    motor: modelInfo && modelInfo.motor_model ? !modelInfo.motor_model.load_error : null,
    gait: true,
  };

  const stepState = loading.active ? 2 : (result ? 3 : 1);

  const exportJSON = () => {
    const payload = {
      generated_at: new Date().toISOString(),
      fusion,
      modality_results: modalityResults,
      latest_result: result,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `parkinsons_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const printReport = () => {
    window.print();
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors duration-300">
      <Header
        theme={theme}
        onToggleTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        demoMode={demoMode}
        onToggleDemo={() => setDemoMode((prev) => !prev)}
        systemStatus={systemStatus}
      />

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        <Card gradient className="border-slate-200/60 dark:border-slate-700/40">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div>
              <h2 className="text-2xl font-semibold text-slate-900 dark:text-white font-display">Multi-Modal Detection System</h2>
              <p className="text-sm text-slate-500 dark:text-slate-300 mt-2">Upload handwriting, motor features, or gait signals for AI-powered assessment across three validated modalities.</p>
              <div className="mt-4 flex flex-wrap gap-2 text-xs">
                <span className="px-3 py-1 rounded-full bg-indigo-100/60 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-200">Research-grade</span>
                <span className="px-3 py-1 rounded-full bg-emerald-100/60 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-200">Multi-modal fusion</span>
                <span className="px-3 py-1 rounded-full bg-slate-100/60 text-slate-600 dark:bg-slate-800/60 dark:text-slate-300">Non-diagnostic demo</span>
              </div>
            </div>
            <div className="glass-panel p-4 rounded-2xl border border-slate-200/60 dark:border-slate-700/50">
              <div className="text-xs uppercase tracking-wide text-slate-400">Workflow</div>
              <div className="mt-3 flex items-center gap-4">
                {['Input', 'Processing', 'Results'].map((label, idx) => {
                  const step = idx + 1;
                  const active = stepState === step;
                  const complete = stepState > step;
                  return (
                    <div key={label} className="flex items-center gap-2">
                      <div className={`h-8 w-8 rounded-full flex items-center justify-center text-sm font-semibold ${active ? 'bg-indigo-600 text-white shadow-lg' : complete ? 'bg-emerald-500 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-500'}`}>
                        {step}
                      </div>
                      <span className={`text-xs font-medium ${active ? 'text-indigo-600 dark:text-indigo-300' : 'text-slate-400'}`}>{label}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </Card>

        <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <section id="handwriting">
              <Card>
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl text-white shadow-lg">
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold">Handwriting Analysis</h3>
                      <p className="text-sm text-slate-500 dark:text-slate-300 mt-1">Upload handwriting samples for deep-learning pattern recognition.</p>
                    </div>
                  </div>
                  <span className="text-xs px-3 py-1 rounded-full bg-slate-100/70 text-slate-600 dark:bg-slate-800/70 dark:text-slate-300">Image · CNN</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <div className="space-y-4">
                    <input
                      className="block w-full text-sm text-slate-600 file:mr-4 file:py-2.5 file:px-5 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 file:cursor-pointer file:transition-all cursor-pointer no-print"
                      type="file"
                      accept="image/*"
                      onChange={handleHandwritingChange}
                    />
                    {fileErrors.handwriting && (
                      <div className="text-xs text-rose-600 bg-rose-50 dark:bg-rose-500/10 dark:text-rose-200 rounded-lg p-2">
                        {fileErrors.handwriting}
                      </div>
                    )}
                    {hwFile && (
                      <div className="text-xs text-slate-500 dark:text-slate-300 bg-slate-100/70 dark:bg-slate-800/50 p-3 rounded-lg">
                        <div>Selected: <span className="font-medium text-slate-700 dark:text-slate-100">{hwFile.name}</span></div>
                        <div className="mt-1">Size: {(hwFile.size / 1024 / 1024).toFixed(2)} MB</div>
                        {hwMeta && <div>Resolution: {hwMeta.width} × {hwMeta.height}</div>}
                      </div>
                    )}
                    <button
                      className="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white rounded-lg font-medium transition-all duration-200 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 no-print"
                      onClick={handleHandwriting}
                      disabled={loading.active}
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      Analyze Handwriting
                    </button>
                  </div>
                  <div className="space-y-3">
                    <div className="text-xs uppercase tracking-wide text-slate-400">Preview</div>
                    <div className="rounded-xl border border-slate-200/60 dark:border-slate-700/50 p-3 bg-white/60 dark:bg-slate-900/50">
                      {hwPreview ? (
                        <img src={hwPreview} alt="Handwriting preview" className="w-full h-48 object-contain rounded-lg" />
                      ) : (
                        <div className="h-48 flex items-center justify-center text-sm text-slate-400">
                          No preview available
                        </div>
                      )}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-300">
                      Tip: Use high-contrast scans and avoid heavy compression artifacts.
                    </div>
                  </div>
                </div>
              </Card>
            </section>

            <section id="motor">
              <Card>
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl text-white shadow-lg">
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold">Motor Features</h3>
                      <p className="text-sm text-slate-500 dark:text-slate-300 mt-1">Structured feature vectors from motor assessment tasks.</p>
                    </div>
                  </div>
                  <span className="text-xs px-3 py-1 rounded-full bg-slate-100/70 text-slate-600 dark:bg-slate-800/70 dark:text-slate-300">Structured · ML</span>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <label className="text-sm font-semibold">JSON Input</label>
                    <textarea
                      className={`w-full p-4 border-2 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all font-mono text-xs resize-none hover:border-gray-300 ${motorJsonValid ? 'border-slate-200 dark:border-slate-700' : 'border-rose-300'}`}
                      rows={9}
                      value={motorJson}
                      onChange={(e) => {
                        setMotorJson(e.target.value);
                        setFileErrors((prev) => ({ ...prev, motorJson: null }));
                      }}
                    />
                    <input
                      className="block w-full text-sm text-slate-600 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 file:cursor-pointer file:transition-all cursor-pointer no-print"
                      type="file"
                      accept=".json,application/json,text/json"
                      onChange={handleMotorJsonFileChange}
                    />
                    {fileErrors.motorJson && (
                      <div className="text-xs text-rose-600 bg-rose-50 dark:bg-rose-500/10 dark:text-rose-200 rounded-lg p-2">
                        {fileErrors.motorJson}
                      </div>
                    )}
                    {motorJsonFile && (
                      <div className="text-xs text-slate-500 dark:text-slate-300 bg-slate-100/70 dark:bg-slate-800/50 p-3 rounded-lg">
                        <div>Selected: <span className="font-medium">{motorJsonFile.name}</span></div>
                        <div className="mt-1">Size: {(motorJsonFile.size / 1024 / 1024).toFixed(2)} MB</div>
                      </div>
                    )}
                    <div className="flex items-center gap-3 text-xs">
                      <span className={`px-2 py-1 rounded-full ${motorJsonValid ? 'bg-emerald-100/60 text-emerald-700' : 'bg-rose-100/60 text-rose-700'}`}>
                        {motorJsonValid ? 'Valid JSON' : 'Invalid JSON'}
                      </span>
                      <button
                        className="text-xs text-indigo-600 dark:text-indigo-300 underline"
                        type="button"
                        onClick={() => setMotorJson(DEMO_MOTOR_JSON)}
                      >
                        Load demo JSON
                      </button>
                    </div>
                    <button
                      className="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white rounded-lg font-medium transition-all duration-200 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 no-print"
                      onClick={handleMotorJson}
                      disabled={loading.active}
                    >
                      Predict from JSON
                    </button>
                  </div>

                  <div className="space-y-3">
                    <label className="text-sm font-semibold">Upload CSV File</label>
                    <input
                      className="block w-full text-sm text-slate-600 file:mr-4 file:py-2.5 file:px-5 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100 file:cursor-pointer file:transition-all cursor-pointer no-print"
                      type="file"
                      accept=".csv,.CSV,text/csv,application/vnd.ms-excel,text/plain"
                      onChange={handleMotorCsvChange}
                    />
                    <button
                      type="button"
                      className="text-xs text-indigo-600 dark:text-indigo-300 underline no-print"
                      onClick={loadDemoMotorCsv}
                    >
                      Use demo CSV
                    </button>
                    {fileErrors.motorCsv && (
                      <div className="text-xs text-rose-600 bg-rose-50 dark:bg-rose-500/10 dark:text-rose-200 rounded-lg p-2">
                        {fileErrors.motorCsv}
                      </div>
                    )}
                    {motorCsvFile && (
                      <div className="text-xs text-slate-500 dark:text-slate-300 bg-slate-100/70 dark:bg-slate-800/50 p-3 rounded-lg">
                        <div>Selected: <span className="font-medium">{motorCsvFile.name}</span></div>
                        <div className="mt-1">Size: {(motorCsvFile.size / 1024 / 1024).toFixed(2)} MB</div>
                      </div>
                    )}
                    {motorCsvPreview && (
                      <pre className="text-xs bg-white/70 dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/50 rounded-lg p-3 max-h-40 overflow-auto">
                        {motorCsvPreview}
                      </pre>
                    )}
                    <button
                      className="px-6 py-2.5 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg font-medium transition-all duration-200 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 no-print"
                      onClick={handleMotorCsv}
                      disabled={loading.active}
                    >
                      Predict from CSV
                    </button>

                    <div className="mt-6 pt-5 border-t border-slate-200/60 dark:border-slate-700/40">
                      <label className="text-sm font-semibold">Upload Timeseries (.txt)</label>
                      <input
                        className="mt-2 block w-full text-sm text-slate-600 file:mr-4 file:py-2.5 file:px-5 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-pink-50 file:text-pink-700 hover:file:bg-pink-100 file:cursor-pointer file:transition-all cursor-pointer no-print"
                        type="file"
                        accept=".txt,.TXT,text/plain,application/octet-stream"
                        multiple
                        onChange={handleMotorTsChange}
                      />
                      <button
                        type="button"
                        className="text-xs text-indigo-600 dark:text-indigo-300 underline mt-2 no-print"
                        onClick={loadDemoMotorTimeseries}
                      >
                        Use demo timeseries
                      </button>
                      {fileErrors.motorTs && (
                        <div className="text-xs text-rose-600 bg-rose-50 dark:bg-rose-500/10 dark:text-rose-200 rounded-lg p-2 mt-2">
                          {fileErrors.motorTs}
                        </div>
                      )}
                      {motorTsFiles.length > 0 && (
                        <div className="text-xs text-slate-500 dark:text-slate-300 bg-slate-100/70 dark:bg-slate-800/50 p-3 rounded-lg mt-2">
                          <div>Selected: <span className="font-medium">{motorTsFiles.length === 1 ? motorTsFiles[0].name : `${motorTsFiles.length} files selected`}</span></div>
                          <div className="mt-1">Size: {(motorTsFiles.reduce((acc, f) => acc + f.size, 0) / 1024 / 1024).toFixed(2)} MB</div>
                        </div>
                      )}
                      {motorTsPreview && (
                        <pre className="text-xs bg-white/70 dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/50 rounded-lg p-3 max-h-40 overflow-auto mt-2">
                          {motorTsPreview}
                        </pre>
                      )}
                      <button
                        className="mt-3 px-6 py-2.5 bg-gradient-to-r from-pink-600 to-red-600 hover:from-pink-700 hover:to-red-700 text-white rounded-lg font-medium transition-all duration-200 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 no-print"
                        onClick={handleMotorTimeseries}
                        disabled={loading.active}
                      >
                        Predict from timeseries
                      </button>
                      <div className="text-xs text-slate-500 dark:text-slate-300 mt-2">
                        Note: you can select multiple .txt files; the backend will aggregate them into one stable patient/not-patient score.
                      </div>
                    </div>

                  </div>
                </div>
              </Card>
            </section>

            <section id="gait">
              <Card>
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-gradient-to-br from-pink-500 to-red-600 rounded-xl text-white shadow-lg">
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold">Gait Analysis</h3>
                      <p className="text-sm text-slate-500 dark:text-slate-300 mt-1">Time-series gait signal analysis for movement irregularities.</p>
                    </div>
                  </div>
                  <span className="text-xs px-3 py-1 rounded-full bg-slate-100/70 text-slate-600 dark:bg-slate-800/70 dark:text-slate-300">Timeseries · Signal</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <div className="space-y-3">
                    <input
                      className="block w-full text-sm text-slate-600 file:mr-4 file:py-2.5 file:px-5 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-pink-50 file:text-pink-700 hover:file:bg-pink-100 file:cursor-pointer file:transition-all cursor-pointer no-print"
                      type="file"
                      accept=".txt,text/plain"
                      onChange={handleGaitChange}
                    />
                    {fileErrors.gait && (
                      <div className="text-xs text-rose-600 bg-rose-50 dark:bg-rose-500/10 dark:text-rose-200 rounded-lg p-2">
                        {fileErrors.gait}
                      </div>
                    )}
                    {gaitFile && (
                      <div className="text-xs text-slate-500 dark:text-slate-300 bg-slate-100/70 dark:bg-slate-800/50 p-3 rounded-lg">
                        <div>Selected: <span className="font-medium">{gaitFile.name}</span></div>
                        <div className="mt-1">Size: {(gaitFile.size / 1024 / 1024).toFixed(2)} MB</div>
                      </div>
                    )}
                    <button
                      className="px-6 py-2.5 bg-gradient-to-r from-pink-600 to-red-600 hover:from-pink-700 hover:to-red-700 text-white rounded-lg font-medium transition-all duration-200 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 no-print"
                      onClick={handleGait}
                      disabled={loading.active}
                    >
                      Analyze Gait
                    </button>
                  </div>
                  <div className="space-y-3">
                    <div className="text-xs uppercase tracking-wide text-slate-400">Signal preview</div>
                    <div className="rounded-xl border border-slate-200/60 dark:border-slate-700/50 p-3 bg-white/60 dark:bg-slate-900/50">
                      {gaitPreview ? (
                        <pre className="text-xs whitespace-pre-wrap text-slate-600 dark:text-slate-300 max-h-40 overflow-auto">{gaitPreview}</pre>
                      ) : (
                        <div className="h-40 flex items-center justify-center text-sm text-slate-400">
                          Upload a gait timeseries (.txt)
                        </div>
                      )}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-300">
                      Format: one value per line or comma-separated values.
                    </div>
                  </div>
                </div>
              </Card>
            </section>

            <Card hover={false}>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold">AI Insight Summary</h3>
                  <p className="text-sm text-slate-500 dark:text-slate-300">Dynamic narrative for clinical transparency (non-diagnostic).</p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs px-3 py-1 rounded-full bg-slate-100/70 text-slate-600 dark:bg-slate-800/70 dark:text-slate-300">Clinical Transparency</span>
                  {fusionBand && (
                    <span className="text-xs px-3 py-1 rounded-full border border-slate-200/60 dark:border-slate-700/50 bg-white/70 dark:bg-slate-900/60 text-slate-700 dark:text-slate-200">
                      Risk: <span className={fusionBand.className}>{fusionBand.label}</span>
                    </span>
                  )}
                </div>
              </div>

              <div className="mt-4 space-y-3">
                <div className="text-sm font-medium text-slate-800 dark:text-slate-100">{clinicalInsight.title}</div>
                <p className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed">{clinicalInsight.text}</p>

                {fusion && (
                  <div className="flex flex-wrap gap-2">
                    {fusion.parts.map((p) => {
                      const ci = modalityResults[p.key]?.confidence_interval || modalityResults[p.key]?.reasoning?.confidence_interval;
                      const badge = badgeForConfidence(ci);
                      const stdTxt = (ci && typeof ci.std === 'number' && Number.isFinite(ci.std)) ? `σ=${ci.std.toFixed(3)}` : null;
                      return (
                        <span key={p.key} className="text-xs px-3 py-1 rounded-full border border-slate-200/60 dark:border-slate-700/50 bg-white/70 dark:bg-slate-900/60 text-slate-700 dark:text-slate-200">
                          {p.label}: <span className="font-semibold">{formatPct(p.score)}</span>
                          <span className="ml-2">
                            <span className={badge.className}>{badge.label}{stdTxt ? ` · ${stdTxt}` : ''}</span>
                          </span>
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="mt-5 pt-4 border-t border-slate-200/60 dark:border-slate-700/50">
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  <span className="font-semibold">Medical disclaimer:</span> This tool is for research/demonstration only and does not provide a medical diagnosis. Interpret results with clinician oversight and standardized data collection.
                </div>
              </div>
            </Card>

            <Card hover={false}>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold">Explainable AI Panel</h3>
                  <p className="text-sm text-slate-500 dark:text-slate-300">Visual + feature evidence supporting each modality output.</p>
                </div>
                <span className="text-xs px-3 py-1 rounded-full bg-slate-100/70 text-slate-600 dark:bg-slate-800/70 dark:text-slate-300">XAI</span>
              </div>

              <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4 text-sm">
                <div className="bg-white/70 dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/50 rounded-xl p-4 xai-fade-in">
                  <div className="flex items-center justify-between">
                    <div className="text-xs uppercase tracking-wide text-slate-400">Handwriting · Grad-CAM</div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500 dark:text-slate-300">Overlay</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={Math.round(hwHeatmapOpacity * 100)}
                        onChange={(e) => setHwHeatmapOpacity(Number(e.target.value) / 100)}
                        className="w-28"
                      />
                    </div>
                  </div>

                  <div className="mt-3">
                    {(hwPreview && (modalityResults.handwriting?.reasoning?.heatmap_url || modalityResults.handwriting?.heatmap_url)) ? (
                      <div className="heatmap-wrap">
                        <img src={hwPreview} alt="handwriting" />
                        <img
                          className="heatmap-overlay"
                          src={resolveAssetUrl(modalityResults.handwriting?.reasoning?.heatmap_url || modalityResults.handwriting?.heatmap_url)}
                          alt="gradcam"
                          style={{ opacity: hwHeatmapOpacity }}
                        />
                      </div>
                    ) : (
                      <div className="text-sm text-slate-500 dark:text-slate-300">Run handwriting analysis with XAI enabled to generate a heatmap overlay.</div>
                    )}
                  </div>

                  <div className="mt-3 text-xs text-slate-500 dark:text-slate-300">
                    Highlighted regions indicate areas most influential to the model output (best-effort; not a diagnostic marker).
                  </div>
                </div>

                <div className="bg-white/70 dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/50 rounded-xl p-4 xai-fade-in">
                  <div className="text-xs uppercase tracking-wide text-slate-400">Fusion · Waterfall</div>
                  <div className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                    Modality push/pull around each threshold (positive pushes toward patient; negative toward healthy).
                  </div>
                  <div className="mt-4 space-y-3">
                    {fusionContributions.length === 0 && (
                      <div className="text-sm text-slate-400">Run at least one modality to see fusion contributions.</div>
                    )}
                    {fusionContributions.map((c) => {
                      const val = c.contribution;
                      const abs = Math.min(1, Math.abs(val) * 3.0);
                      const isPos = val >= 0;
                      return (
                        <div key={c.key} className="space-y-1">
                          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-300">
                            <span className="font-medium">{c.label}</span>
                            <span>{val >= 0 ? '+' : ''}{val.toFixed(3)}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="xai-bar-track flex-1">
                              <div
                                className="xai-bar-fill"
                                style={{
                                  width: `${abs * 100}%`,
                                  background: isPos ? 'rgba(244,63,94,0.85)' : 'rgba(16,185,129,0.85)',
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="bg-white/70 dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/50 rounded-xl p-4 xai-fade-in">
                  <div className="text-xs uppercase tracking-wide text-slate-400">Motor · Feature Importance</div>
                  <div className="mt-3 space-y-3">
                    {(modalityResults.motor?.reasoning?.top_features || []).slice(0, 8).map((f, idx, arr) => {
                      const max = Math.max(...arr.map((x) => x.importance || 0), 1e-9);
                      const pct = ((f.importance || 0) / max) * 100;
                      return (
                        <div key={idx}>
                          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-300">
                            <span className="font-medium">{prettyFeatureName(f.name)}</span>
                            <span>{(f.value != null && Number.isFinite(f.value)) ? f.value.toFixed(3) : '—'}</span>
                          </div>
                          <div className="xai-bar-track mt-1">
                            <div className="xai-bar-fill" style={{ width: `${pct}%`, background: 'rgba(99,102,241,0.85)' }} />
                          </div>
                        </div>
                      );
                    })}
                    {(!modalityResults.motor || !(modalityResults.motor?.reasoning?.top_features || []).length) && (
                      <div className="text-sm text-slate-400">Run a motor analysis to view feature importance.</div>
                    )}
                  </div>
                  <div className="mt-3 text-xs text-slate-500 dark:text-slate-300">Importance values are global (model-level). For true per-sample SHAP/LIME, enable a TreeSHAP backend (optional).</div>
                </div>

                <div className="bg-white/70 dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/50 rounded-xl p-4 xai-fade-in">
                  <div className="text-xs uppercase tracking-wide text-slate-400">Gait · Feature Importance</div>
                  <div className="mt-3 space-y-3">
                    {(modalityResults.gait?.reasoning?.top_features || []).slice(0, 8).map((f, idx, arr) => {
                      const max = Math.max(...arr.map((x) => x.importance || 0), 1e-9);
                      const pct = ((f.importance || 0) / max) * 100;
                      return (
                        <div key={idx}>
                          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-300">
                            <span className="font-medium">{prettyFeatureName(f.name)}</span>
                            <span>{(f.value != null && Number.isFinite(f.value)) ? f.value.toFixed(3) : '—'}</span>
                          </div>
                          <div className="xai-bar-track mt-1">
                            <div className="xai-bar-fill" style={{ width: `${pct}%`, background: 'rgba(236,72,153,0.85)' }} />
                          </div>
                        </div>
                      );
                    })}
                    {(!modalityResults.gait || !(modalityResults.gait?.reasoning?.top_features || []).length) && (
                      <div className="text-sm text-slate-400">Run a gait analysis to view feature importance.</div>
                    )}
                  </div>
                  <div className="mt-3 text-xs text-slate-500 dark:text-slate-300">Top gait features correspond to enhanced stats (COP, step cadence, spectral energy) computed from the uploaded signal.</div>
                </div>
              </div>
            </Card>

            <Card>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold">Timeline of Recent Tests</h3>
                  <p className="text-sm text-slate-500 dark:text-slate-300">Track repeated tests across modalities.</p>
                </div>
                <button
                  className="text-xs text-indigo-600 dark:text-indigo-300 underline no-print"
                  onClick={() => setHistory([])}
                >
                  Clear history
                </button>
              </div>
              <div className="mt-4 space-y-3">
                {history.length === 0 && (
                  <div className="text-sm text-slate-400">No historical results yet.</div>
                )}
                {history.slice(0, 6).map((item) => (
                  <div key={item.id} className="flex items-center justify-between bg-white/70 dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/50 rounded-lg p-3">
                    <div>
                      <div className="text-sm font-medium text-slate-700 dark:text-slate-200">{item.modality}</div>
                      <div className="text-xs text-slate-400">{new Date(item.ts).toLocaleString()}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-semibold text-slate-700 dark:text-slate-200">{item.label || 'n/a'}</div>
                      <div className="text-xs text-slate-400">{item.prob != null ? formatPct(item.prob) : 'n/a'}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          <aside className="space-y-6">
            <Card hover={false}>
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold">Results & Fusion</h4>
                <span className="text-xs text-slate-400">Realtime</span>
              </div>

              {loading.active && (
                <div className="mb-4">
                  <div className="flex items-center gap-2 text-sm text-indigo-600 dark:text-indigo-300">
                    <span className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse"></span>
                    {loading.stage} ({loading.modality})
                  </div>
                  <div className="mt-2 h-2 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 animate-progress"></div>
                  </div>
                </div>
              )}

              {!fusion ? (
                <div className="text-sm text-slate-400">Run at least one modality to see the fusion output.</div>
              ) : (
                <div className="space-y-4">
                  <div className="p-4 rounded-xl border border-slate-200/60 dark:border-slate-700/50 bg-white/80 dark:bg-slate-900/70">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-xs uppercase tracking-wide text-slate-400">Fusion Result</div>
                        <div className="text-lg font-semibold text-slate-800 dark:text-white">Risk: <span className={fusionBand.className}>{fusionBand.label}</span></div>
                        <div className="text-xs text-slate-500 dark:text-slate-300">{fusion.agreement}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-slate-400">Score</div>
                        <div className="text-2xl font-bold text-slate-800 dark:text-white">{formatPct(fusion.score)}</div>
                        <div className={`text-xs font-semibold ${confidenceLabel === 'Low' ? 'text-amber-600' : (confidenceLabel === 'Medium' ? 'text-yellow-500' : 'text-emerald-500')}`}>{confidenceLabel} confidence</div>
                      </div>
                    </div>
                    <div className="mt-4">
                      <div className="h-2 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden relative">
                        <div className="h-full bg-gradient-to-r from-emerald-400 via-amber-400 to-rose-500" style={{ width: `${(fusion.score || 0) * 100}%` }}></div>
                        <div className="absolute top-[-4px] w-1 h-4 bg-slate-800 dark:bg-slate-100" style={{ left: '50%' }} title="Decision midpoint"></div>
                      </div>
                      <div className="mt-2 text-xs text-slate-400">Fusion combines available modalities with weights (Handwriting 0.4, Motor 0.3, Gait 0.3).</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {fusion.parts.map((p) => (
                      <div key={p.key} className="p-3 rounded-lg border border-slate-200/60 dark:border-slate-700/50 bg-white/70 dark:bg-slate-900/60">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-medium">{p.label}</div>
                          <div className="text-sm text-slate-600 dark:text-slate-300">{formatPct(p.score)}</div>
                        </div>
                        <div className="mt-2 h-2 bg-slate-200 dark:bg-slate-800 rounded-full relative">
                          <div className="h-full bg-indigo-500" style={{ width: `${p.score * 100}%` }}></div>
                          <div className="absolute top-[-3px] w-1 h-3 bg-amber-500" style={{ left: `${p.threshold * 100}%` }} title="Threshold"></div>
                        </div>
                        <div className="mt-1 text-xs text-slate-400">Threshold: {p.threshold.toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </Card>

            <Card hover={false}>
              <h4 className="font-semibold mb-3">Latest Result</h4>
              {!result ? (
                <div className="text-sm text-slate-400">No predictions yet.</div>
              ) : result.error ? (
                <div className="bg-rose-50 border border-rose-200 text-rose-700 dark:bg-rose-500/10 dark:border-rose-500/30 dark:text-rose-200 rounded-lg p-3 text-sm">
                  {result.error}
                </div>
              ) : (
                <div className="space-y-3 text-sm">
                  <div className="bg-white/80 dark:bg-slate-900/70 border border-slate-200/60 dark:border-slate-700/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400">Endpoint</div>
                    <div className="font-semibold text-slate-700 dark:text-slate-200">{result.endpoint}</div>
                  </div>
                  <div className="bg-white/80 dark:bg-slate-900/70 border border-slate-200/60 dark:border-slate-700/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400">Label</div>
                    <div className="font-semibold text-slate-700 dark:text-slate-200">{result.data && result.data.label ? String(result.data.label) : 'n/a'}</div>
                  </div>
                  {result.data && result.data.prob_pos != null && (
                    <div className="bg-white/80 dark:bg-slate-900/70 border border-slate-200/60 dark:border-slate-700/50 rounded-lg p-3">
                      <div className="text-xs text-slate-400">Probability (patient)</div>
                      <div className="font-semibold text-slate-700 dark:text-slate-200">{formatPct(result.data.prob_pos)}</div>
                    </div>
                  )}
                  {result.endpoint === 'handwriting' && result.data && (
                    <div className="bg-white/80 dark:bg-slate-900/70 border border-slate-200/60 dark:border-slate-700/50 rounded-lg p-3 text-xs space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Threshold</span>
                        <span className="text-slate-600 dark:text-slate-200">{result.data.threshold != null ? result.data.threshold.toFixed(3) : 'n/a'}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">ResNet50 prob</span>
                        <span className="text-slate-600 dark:text-slate-200">{result.data.prob_a != null ? result.data.prob_a.toFixed(4) : 'n/a'}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">EffNetB2 prob</span>
                        <span className="text-slate-600 dark:text-slate-200">{result.data.prob_b != null ? result.data.prob_b.toFixed(4) : 'n/a'}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </Card>

            <Card>
              <h4 className="font-semibold mb-3">Model Transparency</h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500 dark:text-slate-300">Handwriting Models</span>
                  <span className="text-xs text-slate-600 dark:text-slate-200">{handwritingModelNames}</span>
                </div>
                {handwritingMeta && (
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500 dark:text-slate-300">Threshold</span>
                    <span className="text-xs text-slate-600 dark:text-slate-200">{handwritingMeta.threshold != null ? handwritingMeta.threshold.toFixed(3) : 'n/a'}</span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-slate-500 dark:text-slate-300">Backend Status</span>
                  <span className="text-xs text-emerald-600">Online</span>
                </div>
              </div>
            </Card>

            <Card>
              <h4 className="font-semibold mb-3">Export Report</h4>
              <div className="space-y-3">
                <button
                  className="w-full px-4 py-2 rounded-lg bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 font-semibold shadow-sm hover:shadow-md transition-all no-print"
                  onClick={exportJSON}
                >
                  Download JSON
                </button>
                <button
                  className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-100 font-semibold hover:bg-slate-100 dark:hover:bg-slate-800 transition-all no-print"
                  onClick={printReport}
                >
                  Print / Save as PDF
                </button>
                <p className="text-xs text-slate-400">Exports include fusion, modality details, and timestamps.</p>
              </div>
            </Card>

            <Card>
              <h4 className="font-semibold mb-3">Quick Guide</h4>
              <div className="space-y-2 text-sm text-slate-500 dark:text-slate-300">
                <div>Use clear scans on plain backgrounds.</div>
                <div>Motor JSON must include all features used in training.</div>
                <div>Gait signals should be raw timeseries (.txt).</div>
              </div>
            </Card>
          </aside>
        </main>
      </div>
    </div>
  );
}

export default App;
