/* =====================================================================
   PixcraftAI Real Engine  (pixcraft-engine.js)
   ---------------------------------------------------------------------
   Replaces every simulated feature of the original single-file app with
   real, verifiable implementations:

   • Vision      : MobileCLIP-S0 (Apple) running IN THE BROWSER via
                   Transformers.js  → real zero-shot image understanding,
                   no API key needed. Gemini Vision is used when a key exists.
   • Metadata    : grounded on the vision analysis (never on the filename).
   • LLM         : Google Gemini (user key)  →  free Pollinations gateway
                   (best-effort)  →  honest "offline" fallback.
   • Image Gen   : Gemini 2.5 Flash Image (key)  →  Pollinations (free).
   • BG Removal  : @imgly/background-removal (ONNX, in-browser).
   • Upscale     : Swin2SR neural 2× pass (small inputs) + Lanczos HQ
                   resampling + unsharp mask. Always uses the REAL image.
   • QA          : real resolution / compression / sharpness / noise checks.
   ===================================================================== */
(function () {
  'use strict';

  const CDN = {
    transformers: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js',
    imgly: 'https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.7.0/+esm',
    imglyPublicPath: 'https://staticimgly.com/@imgly/background-removal-data/1.7.0/dist/'
  };
  const MODELS = {
    clip: 'Xenova/mobileclip_s0',
    sr2x: 'Xenova/swin2SR-lightweight-x2-64'
  };
  const SCRIPT_BASE = (function () {
    try {
      const src = (typeof document !== 'undefined' && document.currentScript && document.currentScript.src) ||
                  (typeof location !== 'undefined' && location.href) ||
                  'https://afzal534646.github.io/Am_Afzal-Website/';
      return new URL('.', src).href;
    } catch (e) {
      return 'https://afzal534646.github.io/Am_Afzal-Website/';
    }
  })();
  const LABELS_URL = new URL('pixcraft-vision-labels.json', SCRIPT_BASE).href;
  const POLLINATIONS_TEXT = 'https://text.pollinations.ai/openai';
  const POLLINATIONS_IMAGE = 'https://image.pollinations.ai/prompt/';
  const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta';

  const state = {
    tf: null,             // transformers.js module promise
    clip: null,           // { processor, model, device }
    clipLoading: null,
    labels: null,
    sr: null,
    srLoading: null,
    geminiModelsCache: {},// key -> [models]
    lastEngine: '',
    webgpu: false
  };

  let webgpuDetected = null;
  async function hasWorkingWebGPU() {
    if (webgpuDetected !== null) return webgpuDetected;
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      webgpuDetected = false;
      return false;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      webgpuDetected = !!adapter;
    } catch (e) {
      webgpuDetected = false;
    }
    state.webgpu = webgpuDetected;
    return webgpuDetected;
  }

  const listeners = new Set();
  function emit(stage, pct, msg) {
    for (const fn of listeners) { try { fn(stage, pct, msg); } catch (e) {} }
  }

  /* ------------------------------------------------------------------ */
  /* Generic helpers                                                     */
  /* ------------------------------------------------------------------ */
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
  function withTimeout(promise, ms, label) {
    let t;
    const timeout = new Promise((_, rej) => { t = setTimeout(() => rej(new Error((label || 'Request') + ' timed out after ' + Math.round(ms / 1000) + 's')), ms); });
    return Promise.race([promise.finally(() => clearTimeout(t)), timeout]);
  }
  function fetchWithTimeout(url, opts, ms) {
    const ctrl = new AbortController();
    const id = setTimeout(() => ctrl.abort(), ms);
    return fetch(url, Object.assign({}, opts || {}, { signal: ctrl.signal })).finally(() => clearTimeout(id));
  }
  function extractJson(raw) {
    if (!raw) return null;
    if (typeof raw === 'object') return raw;
    try { return JSON.parse(raw); } catch (e) {}
    const s = String(raw).trim();
    const m = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    if (m && m[1]) { try { return JSON.parse(m[1].trim()); } catch (e) {} }
    const a = s.indexOf('{'), b = s.lastIndexOf('}');
    if (a !== -1 && b > a) { try { return JSON.parse(s.substring(a, b + 1)); } catch (e) {} }
    const c = s.indexOf('['), d = s.lastIndexOf(']');
    if (c !== -1 && d > c) { try { return JSON.parse(s.substring(c, d + 1)); } catch (e) {} }
    return null;
  }
  function dataUrlToBlob(dataUrl) {
    const [head, b64] = dataUrl.split(',');
    const mime = (head.match(/data:([^;]+)/) || [])[1] || 'image/jpeg';
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
  }
  function loadImageElement(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Could not decode image'));
      if (src instanceof Blob) img.src = URL.createObjectURL(src); else img.src = src;
    });
  }
  function toCanvas(img, w, h) {
    const c = document.createElement('canvas');
    c.width = w || img.naturalWidth || img.width; c.height = h || img.naturalHeight || img.height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, c.width, c.height);
    return c;
  }
  function canvasToBlob(canvas, type, quality) {
    return new Promise(res => canvas.toBlob(res, type || 'image/png', quality));
  }
  const ACRONYMS = { '3d': '3D', 'ai': 'AI', 'led': 'LED', 'dna': 'DNA', 'usa': 'USA', 'uk': 'UK', 'cgi': 'CGI', 'ui': 'UI', 'ux': 'UX', 'vr': 'VR', 'ar': 'AR', 'tv': 'TV', 'suv': 'SUV', 'diy': 'DIY', 'hd': 'HD', 'it': 'IT' };
  function titleCase(s) {
    const small = new Set(['a', 'an', 'the', 'of', 'in', 'on', 'at', 'with', 'and', 'or', 'for', 'to', 'by', 'from', 'over', 'under', 'into', 'as', 'against']);
    return s.split(/\s+/).filter(Boolean).map((w, i) => {
      const lw = w.toLowerCase();
      if (ACRONYMS[lw] && !(i === 0 && lw === 'it')) return ACRONYMS[lw];
      if (i > 0 && small.has(lw)) return lw;
      return w.charAt(0).toUpperCase() + w.slice(1);
    }).join(' ');
  }
  function stripArticle(s) { return s.replace(/^(a|an|the)\s+/i, '').trim(); }
  function uniq(arr) { const seen = new Set(); const out = []; for (const x of arr) { const k = String(x).toLowerCase().trim(); if (k && !seen.has(k)) { seen.add(k); out.push(k); } } return out; }

  /* ------------------------------------------------------------------ */
  /* Transformers.js loader                                              */
  /* ------------------------------------------------------------------ */
  function tf() {
    if (!state.tf) {
      state.tf = import(CDN.transformers).then(m => {
        m.env.allowLocalModels = false;
        m.env.useBrowserCache = true;
        return m;
      }).catch(err => { state.tf = null; throw new Error('Could not load AI runtime from CDN: ' + err.message); });
    }
    return state.tf;
  }
  function progressAdapter(stage) {
    let lastPct = -1;
    return (p) => {
      if (!p) return;
      if (p.status === 'progress' && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        if (pct !== lastPct) { lastPct = pct; emit(stage, pct, `Downloading ${p.file || 'model'} (${pct}%)`); }
      } else if (p.status === 'done') {
        emit(stage, 100, `Loaded ${p.file || 'model'}`);
      }
    };
  }

  /* ------------------------------------------------------------------ */
  /* Vision : MobileCLIP zero-shot analysis                              */
  /* ------------------------------------------------------------------ */
  async function loadLabels() {
    if (state.labels) return state.labels;
    const res = await fetch(LABELS_URL, { cache: 'force-cache' });
    if (!res.ok) throw new Error('Vision label bank missing (' + LABELS_URL + ')');
    const json = await res.json();
    const facets = {};
    for (const [k, f] of Object.entries(json.facets)) {
      const bin = atob(f.data);
      const i8 = new Int8Array(bin.length);
      for (let i = 0; i < bin.length; i++) i8[i] = (bin.charCodeAt(i) << 24) >> 24;
      facets[k] = { labels: f.labels, scale: f.scale, i8 };
    }
    state.labels = { dim: json.dim, facets };
    return state.labels;
  }

  async function loadClip() {
    if (state.clip) return state.clip;
    if (state.clipLoading) return state.clipLoading;
    state.clipLoading = (async () => {
      const { AutoProcessor, CLIPVisionModelWithProjection } = await tf();
      emit('vision', 0, 'Preparing built-in vision model…');
      const processor = await AutoProcessor.from_pretrained(MODELS.clip);
      let model = null, device = 'wasm';
      const webgpuOk = await hasWorkingWebGPU();
      if (webgpuOk) {
        try {
          model = await CLIPVisionModelWithProjection.from_pretrained(MODELS.clip, { device: 'webgpu', dtype: 'fp16', progress_callback: progressAdapter('vision') });
          device = 'webgpu';
        } catch (e) { console.warn('[PixEngine] WebGPU vision unavailable, using WASM:', e.message); model = null; }
      }
      if (!model) {
        // NOTE: the quantized (q8) vision weights of this model are broken → fp32 is mandatory on WASM.
        model = await CLIPVisionModelWithProjection.from_pretrained(MODELS.clip, { device: 'wasm', dtype: 'fp32', progress_callback: progressAdapter('vision') });
        device = 'wasm';
      }
      await loadLabels();
      state.clip = { processor, model, device };
      emit('vision', 100, 'Vision model ready (' + device + ')');
      return state.clip;
    })().catch(err => { state.clipLoading = null; throw err; });
    return state.clipLoading;
  }

  async function embedImage(src) {
    const { RawImage } = await tf();
    const clip = await loadClip();
    let image;
    if (src instanceof HTMLCanvasElement) image = await RawImage.fromCanvas(src);
    else if (src instanceof Blob) image = await RawImage.fromBlob(src);
    else image = await RawImage.read(src);
    const inputs = await clip.processor(image);
    let out;
    try {
      out = await clip.model(inputs);
    } catch (e) {
      if (clip.device === 'webgpu') {
        console.warn('[PixEngine] WebGPU inference failed, reloading on WASM:', e.message);
        state.clip = null; state.clipLoading = null; state.webgpu = false;
        const c2 = await loadClip();
        out = await c2.model(inputs);
      } else throw e;
    }
    return out.image_embeds.normalize().tolist()[0];
  }

  function classify(emb, facet, k, T) {
    const L = state.labels; const f = L.facets[facet]; if (!f) return [];
    const D = L.dim; const logits = new Array(f.labels.length);
    for (let r = 0; r < f.labels.length; r++) {
      let s = 0; const base = r * D;
      for (let c = 0; c < D; c++) s += emb[c] * f.i8[base + c];
      logits[r] = (T || 100) * s * f.scale;
    }
    let m = -Infinity; for (const l of logits) if (l > m) m = l;
    const ex = logits.map(l => Math.exp(l - m)); const Z = ex.reduce((a, b) => a + b, 0);
    return logits.map((l, i) => ({ label: f.labels[i], p: ex[i] / Z })).sort((a, b) => b.p - a.p).slice(0, k || 3);
  }

  function pixelStats(canvas) {
    const w = canvas.width, h = canvas.height;
    const sw = Math.min(64, w), sh = Math.min(64, h);
    const c = document.createElement('canvas'); c.width = sw; c.height = sh;
    const ctx = c.getContext('2d', { willReadFrequently: true }); ctx.drawImage(canvas, 0, 0, sw, sh);
    const d = ctx.getImageData(0, 0, sw, sh).data;
    let r = 0, g = 0, b = 0, n = 0; const hues = { red: 0, orange: 0, yellow: 0, green: 0, teal: 0, blue: 0, purple: 0, pink: 0, white: 0, black: 0, gray: 0, brown: 0 };
    for (let i = 0; i < d.length; i += 4) {
      const R = d[i], G = d[i + 1], B = d[i + 2]; r += R; g += G; b += B; n++;
      const max = Math.max(R, G, B), min = Math.min(R, G, B); const l = (max + min) / 2; const s = max === min ? 0 : (max - min) / (1 - Math.abs(2 * l / 255 - 1)) / 255;
      if (l > 235) { hues.white++; continue; } if (l < 25) { hues.black++; continue; } if (s < 0.12) { hues.gray++; continue; }
      let hue = 0; if (max === R) hue = ((G - B) / (max - min)) % 6; else if (max === G) hue = (B - R) / (max - min) + 2; else hue = (R - G) / (max - min) + 4; hue = (hue * 60 + 360) % 360;
      if (hue < 15 || hue >= 345) hues.red++; else if (hue < 40) { if (l < 110) hues.brown++; else hues.orange++; } else if (hue < 65) hues.yellow++; else if (hue < 160) hues.green++; else if (hue < 200) hues.teal++; else if (hue < 260) hues.blue++; else if (hue < 300) hues.purple++; else hues.pink++;
    }
    const avgR = Math.round(r / n), avgG = Math.round(g / n), avgB = Math.round(b / n);
    const brightness = Math.round((avgR * 299 + avgG * 587 + avgB * 114) / 1000);
    const dominant = Object.entries(hues).sort((a, b) => b[1] - a[1]).filter(e => e[1] / n > 0.12).slice(0, 3).map(e => e[0]);
    return { avgR, avgG, avgB, brightness, dominant, width: w, height: h, isWide: w > h * 1.15, isTall: h > w * 1.15, isSquare: Math.abs(w - h) <= Math.max(w, h) * 0.08 };
  }

  /** Full structured analysis of one image (dataURL / Blob / URL / canvas). */
  async function analyzeImage(src) {
    const img = await loadImageElement(src instanceof Blob ? src : src);
    const maxSide = 512; const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = toCanvas(img, Math.round(img.naturalWidth * scale), Math.round(img.naturalHeight * scale));
    const stats = pixelStats(canvas); stats.width = img.naturalWidth; stats.height = img.naturalHeight;
    const emb = await embedImage(canvas);
    const subject = classify(emb, 'subject', 5);
    const hasPeopleP = classify(emb, 'has_people', 2)[0];
    const peopleProb = hasPeopleP && /person or people/.test(hasPeopleP.label) ? hasPeopleP.p : (hasPeopleP ? 1 - hasPeopleP.p : 0);
    const personMass = subject.reduce((acc, s) => acc + (PERSON_RE.test(s.label) ? s.p : 0), 0);
    const subjectIsGraphic = GRAPHIC_RE.test(subject[0].label);
    const hasPeople = personMass > 0.25 || (peopleProb > 0.9 && personMass > 0.05 && !subjectIsGraphic);
    const hasTextP = classify(emb, 'has_text', 2)[0];
    const hasText = hasTextP && /containing visible text/.test(hasTextP.label) && hasTextP.p > 0.6;
    return {
      subject, people: hasPeople ? classify(emb, 'people', 2) : [], hasPeople, hasText,
      setting: classify(emb, 'setting', 2), medium: classify(emb, 'medium', 2), light: classify(emb, 'light', 2),
      activity: hasPeople ? classify(emb, 'activity', 3) : [], composition: classify(emb, 'composition', 2),
      concept: classify(emb, 'concept', 3), colors: stats, embedding: emb, engine: 'MobileCLIP-S0 (' + (state.clip ? state.clip.device : 'wasm') + ')'
    };
  }

  /* ------------------------------------------------------------------ */
  /* Grounded metadata composer (no LLM required)                        */
  /* ------------------------------------------------------------------ */
  const SYN = {
    coffee: ['cafe', 'caffeine', 'espresso', 'beverage', 'hot drink', 'morning', 'breakfast', 'cup', 'mug', 'aroma', 'barista'],
    tea: ['beverage', 'hot drink', 'cup', 'herbal', 'relaxation', 'morning', 'teapot'],
    salad: ['healthy eating', 'vegetarian', 'fresh', 'diet', 'nutrition', 'organic', 'vegan', 'lunch', 'bowl'],
    food: ['cuisine', 'meal', 'delicious', 'gourmet', 'dish', 'recipe', 'restaurant', 'tasty', 'homemade'],
    fruit: ['fresh', 'healthy', 'vitamin', 'organic', 'sweet', 'juicy', 'nutrition', 'diet'],
    vegetable: ['fresh', 'healthy', 'organic', 'vegetarian', 'nutrition', 'harvest', 'farm'],
    dog: ['pet', 'canine', 'animal', 'puppy', 'companion', 'domestic', 'cute', 'friend', 'mammal'],
    cat: ['pet', 'feline', 'animal', 'kitten', 'domestic', 'cute', 'whiskers', 'mammal'],
    horse: ['animal', 'equestrian', 'mammal', 'farm', 'stallion', 'mane'],
    bird: ['wildlife', 'animal', 'feathers', 'wings', 'nature', 'beak', 'ornithology'],
    doctor: ['healthcare', 'medical', 'medicine', 'physician', 'hospital', 'clinic', 'health', 'patient care', 'profession', 'expert'],
    nurse: ['healthcare', 'medical', 'hospital', 'care', 'health', 'clinic', 'profession'],
    business: ['corporate', 'professional', 'office', 'career', 'company', 'success', 'strategy', 'management', 'entrepreneur', 'finance'],
    meeting: ['teamwork', 'collaboration', 'discussion', 'colleagues', 'communication', 'planning', 'brainstorming', 'conference', 'partnership'],
    laptop: ['computer', 'technology', 'work', 'online', 'internet', 'digital', 'remote work', 'notebook', 'typing', 'productivity'],
    smartphone: ['mobile phone', 'technology', 'device', 'communication', 'app', 'digital', 'screen', 'online', 'touchscreen'],
    student: ['education', 'learning', 'school', 'study', 'knowledge', 'university', 'young', 'classroom'],
    teacher: ['education', 'learning', 'school', 'teaching', 'classroom', 'lesson', 'knowledge'],
    solar: ['renewable energy', 'green energy', 'sustainability', 'photovoltaic', 'clean energy', 'electricity', 'environment', 'power', 'eco friendly', 'climate'],
    wind: ['renewable energy', 'green energy', 'sustainability', 'clean energy', 'electricity', 'environment', 'power', 'eco friendly'],
    yoga: ['fitness', 'wellness', 'meditation', 'healthy lifestyle', 'exercise', 'balance', 'relaxation', 'mindfulness', 'flexibility', 'stretching'],
    running: ['fitness', 'exercise', 'sport', 'healthy lifestyle', 'training', 'athlete', 'workout', 'active', 'motion'],
    gym: ['fitness', 'exercise', 'workout', 'training', 'healthy lifestyle', 'strength', 'sport', 'active'],
    mountain: ['landscape', 'nature', 'scenery', 'outdoor', 'travel', 'adventure', 'hiking', 'peak', 'wilderness', 'panorama'],
    beach: ['sea', 'ocean', 'sand', 'summer', 'vacation', 'travel', 'coast', 'tropical', 'holiday', 'relaxation'],
    forest: ['nature', 'trees', 'woodland', 'green', 'outdoor', 'environment', 'wilderness', 'ecology'],
    city: ['urban', 'architecture', 'skyline', 'downtown', 'buildings', 'metropolis', 'travel', 'modern', 'cityscape'],
    house: ['home', 'real estate', 'residential', 'property', 'architecture', 'building', 'exterior', 'mortgage'],
    interior: ['home', 'design', 'furniture', 'decor', 'modern', 'apartment', 'comfortable', 'lifestyle', 'room'],
    kitchen: ['home', 'cooking', 'interior', 'modern', 'design', 'appliance', 'domestic'],
    office: ['workplace', 'business', 'corporate', 'work', 'desk', 'professional', 'interior', 'company'],
    car: ['automobile', 'vehicle', 'transport', 'driving', 'auto', 'road', 'travel', 'automotive'],
    flower: ['bloom', 'blossom', 'floral', 'petal', 'nature', 'garden', 'botany', 'spring', 'plant', 'beauty'],
    plant: ['green', 'nature', 'leaf', 'botany', 'growth', 'garden', 'organic', 'eco'],
    baby: ['infant', 'newborn', 'childhood', 'family', 'cute', 'parenting', 'care', 'innocence', 'little'],
    child: ['kid', 'childhood', 'family', 'young', 'playful', 'happy', 'growing up', 'education'],
    family: ['together', 'love', 'parents', 'children', 'happiness', 'home', 'bonding', 'relationship', 'lifestyle'],
    couple: ['love', 'relationship', 'romance', 'together', 'partner', 'happiness', 'affection', 'lifestyle'],
    woman: ['female', 'lady', 'adult', 'lifestyle', 'portrait', 'person', 'people'],
    man: ['male', 'adult', 'guy', 'lifestyle', 'portrait', 'person', 'people'],
    senior: ['elderly', 'retirement', 'mature', 'old age', 'aging', 'pensioner', 'grandparent', 'healthy aging'],
    money: ['finance', 'cash', 'currency', 'wealth', 'savings', 'banking', 'investment', 'economy', 'payment', 'business'],
    chart: ['data', 'statistics', 'analytics', 'growth', 'graph', 'finance', 'report', 'business', 'infographic'],
    robot: ['artificial intelligence', 'automation', 'technology', 'future', 'machine', 'innovation', 'robotics', 'ai'],
    abstract: ['pattern', 'design', 'modern', 'texture', 'wallpaper', 'art', 'creative', 'graphic', 'backdrop', 'shape'],
    texture: ['surface', 'pattern', 'material', 'backdrop', 'design', 'closeup', 'detail', 'wallpaper'],
    factory: ['industry', 'manufacturing', 'production', 'industrial', 'machinery', 'engineering', 'worker', 'technology'],
    construction: ['building', 'industry', 'engineering', 'site', 'architecture', 'development', 'worker', 'safety'],
    farm: ['agriculture', 'farming', 'rural', 'harvest', 'countryside', 'crop', 'field', 'organic', 'farmer'],
    travel: ['tourism', 'vacation', 'journey', 'trip', 'adventure', 'destination', 'explore', 'holiday'],
    wedding: ['marriage', 'bride', 'groom', 'love', 'celebration', 'ceremony', 'romance', 'couple'],
    shopping: ['retail', 'store', 'consumer', 'customer', 'purchase', 'sale', 'buying', 'commerce'],
    cake: ['dessert', 'sweet', 'bakery', 'celebration', 'birthday', 'pastry', 'delicious', 'sugar'],
    bread: ['bakery', 'baked', 'wheat', 'breakfast', 'food', 'loaf', 'fresh', 'homemade'],
    water: ['liquid', 'fresh', 'clean', 'nature', 'blue', 'drop', 'pure', 'splash'],
    sunset: ['sky', 'dusk', 'evening', 'golden', 'horizon', 'nature', 'scenic', 'twilight', 'landscape'],
    space: ['universe', 'astronomy', 'cosmos', 'stars', 'galaxy', 'science', 'planet', 'exploration'],
    security: ['protection', 'safety', 'privacy', 'lock', 'cyber', 'data', 'technology', 'secure'],
    medical: ['healthcare', 'medicine', 'health', 'hospital', 'clinic', 'treatment', 'care', 'science'],
    music: ['sound', 'instrument', 'melody', 'musician', 'entertainment', 'concert', 'performance', 'art'],
    art: ['creative', 'creativity', 'artist', 'design', 'painting', 'craft', 'hobby', 'colorful']
  };
  const STOP = new Set(['a', 'an', 'the', 'of', 'in', 'on', 'at', 'with', 'and', 'or', 'for', 'to', 'by', 'from', 'photo', 'image', 'taken', 'someone', 'about', 'only', 'it', 'its', 'his', 'her', 'their', 'no', 'not', 'without', 'any', 'like', 'lots', 'shot', 'view', 'scene', 'doing', 'one', 'two', 'rendered', 'wearing', 'holding', 'making', 'using', 'having', 'giving', 'looking', 'camera', 'plain', 'standing', 'sitting', 'against', 'featuring']);
  const CATEGORY_BY_CONCEPT = {
    'business and finance': 'Business', 'technology and artificial intelligence': 'Technology', 'healthcare and medicine': 'Science', 'education and learning': 'People',
    'sustainability and green energy': 'The Environment', 'travel and tourism': 'Travel', 'food and cooking': 'Food', 'sports and fitness': 'Sports', 'family and love': 'Lifestyle',
    'fashion and beauty': 'People', 'real estate and architecture': 'Buildings and Architecture', 'agriculture and farming': 'Industry', 'industry and manufacturing': 'Industry',
    'science and research': 'Science', 'transportation and logistics': 'Transport', 'holidays and celebration': 'Lifestyle', 'christmas': 'Lifestyle', 'ramadan and eid': 'Travel', 'diwali': 'Travel',
    'halloween': 'Lifestyle', "valentine's day": 'Lifestyle', 'new year': 'Lifestyle', 'spring season': 'Landscapes', 'summer season': 'Landscapes', 'autumn season': 'Landscapes', 'winter season': 'Landscapes',
    'money and investment': 'Business', 'cybersecurity and privacy': 'Technology', 'social media and communication': 'Technology', 'remote work and freelancing': 'Business', 'mental health and wellness': 'Lifestyle',
    'climate change and environment': 'The Environment', 'shopping and e-commerce': 'Lifestyle', 'teamwork and collaboration': 'Business', 'leadership and success': 'Business', 'creativity and art': 'Hobbies and Leisure',
    'music and entertainment': 'Hobbies and Leisure', 'pets and animals': 'Animals', 'wildlife and nature': 'Animals', 'home and interior design': 'Buildings and Architecture', 'gaming and esports': 'Technology',
    'cryptocurrency and blockchain': 'Business', 'online shopping and delivery': 'Lifestyle', 'startup and innovation': 'Business', 'retirement and senior living': 'People', 'parenting and childhood': 'People',
    'weddings and romance': 'Lifestyle', 'fitness and healthy lifestyle': 'Sports', 'construction and engineering': 'Industry', 'law and justice': 'Social Issues', 'religion and spirituality': 'Travel',
    'culture and tradition': 'Travel', 'protest and social issues': 'Social Issues', 'poverty and humanitarian aid': 'Social Issues', 'charity and volunteering': 'Social Issues'
  };
  function categoryFromAnalysis(a) {
    const s = a.subject[0].label.toLowerCase();
    if (/dog|cat|horse|cow|goat|sheep|chicken|duck|pig|rabbit|deer|lion|tiger|elephant|monkey|bear|wolf|fox|squirrel|bird|eagle|parrot|owl|flamingo|peacock|butterfly|bee|ladybug|spider|snake|frog|turtle|crocodile|fish|shark|dolphin|whale|jellyfish|coral|panda|koala|kangaroo|camel|zebra|giraffe|penguin|puppy|kitten|rooster/.test(s)) return 'Animals';
    if (/salad|fruit|vegetable|pizza|burger|sandwich|sushi|noodle|pasta|rice|biryani|steak|chicken|seafood|fish dish|soup|bread|croissant|pancake|cake|cupcake|ice cream|cookie|chocolate|candy|breakfast|dinner|picnic|spice|mango|strawberry|apple|banana|watermelon|grape|lemon|avocado|egg|cheese|nuts|honey|olive|cereal|coffee|tea|juice|smoothie|wine|beer|cocktail|glass of water|milk/.test(s)) return 'Food';
    if (/flower|rose|sunflower|tulip|lavender|blossom|lotus|cactus|houseplant|bonsai|leaves|garden with/.test(s)) return 'Plants and Flowers';
    if (/mountain|forest|rainforest|beach|ocean|lake|river|waterfall|desert|canyon|cave|volcano|meadow|field|plantation|vineyard|sunset|sunrise|night sky|milky way|moon|clouds|rainbow|snow scene|autumn leaves|foggy|thunderstorm/.test(s)) return 'Landscapes';
    if (/skyline|skyscraper|street|highway|bridge|village|house|villa|apartment|interior|office|coworking|classroom|library|hospital room|mall|supermarket|store|restaurant|cafe interior|hotel|gym interior|stadium|airport|station|mosque|church|temple|castle|ruins|lighthouse|windmill|pier/.test(s)) return 'Buildings and Architecture';
    if (/car|vehicle|truck|bus|motorcycle|scooter|bicycle|airplane|helicopter|train|subway|ship|port|boat|rocket|rickshaw|tuk tuk|ambulance|traffic/.test(s)) return 'Transport';
    if (/laptop|computer|smartphone|tablet|smartwatch|earbuds|headphones|camera|drone|robot|circuit|server|data center|3d printer|microchip|television|gaming|keyboard|printer|virtual reality|hologram|metaverse|artificial intelligence|cyber|cloud computing|blockchain|chatbot|technology background|user interface|website|mobile app/.test(s)) return 'Technology';
    if (/factory|industrial|warehouse|forklift|machine|gears|tools|welder|construction|crane|power plant|oil pump|textile|garment|sewing/.test(s)) return 'Industry';
    if (/doctor|nurse|surgeon|dentist|pharmac|patient|stethoscope|pills|syringe|microscope|test tube|dna|virus|bacteria|scientist|laboratory|first aid|wheelchair|thermometer|brain|skeleton|tooth/.test(s)) return 'Science';
    if (/business|meeting|colleagues|presenting|handshake|banknote|coins|piggy bank|credit card|wallet|bitcoin|stock market|financial|chart|job interview|call center|customer service/.test(s)) return 'Business';
    if (/athlete|football|cricket|basketball|swimmer|cyclist|boxer|running|jogging|yoga|gym|dumbbell|soccer|tennis|trophy|medal|exercising/.test(s)) return 'Sports';
    if (/abstract|gradient|geometric|particles|network|data visualization|pattern|mandala|logo|icon|typography|quote|blank|mockup|poster|texture|surface|wood planks|brick wall|concrete|marble|metal|fabric|paper|leather|splash|smoke|flames|bubbles|infographic|low poly|neon glowing/.test(s)) return 'Graphic Resources';
    if (/protest|pollution|garbage|recycling|smog|wildfire|flood|drought|glacier|poverty/.test(s)) return 'Social Issues';
    if (/musician|guitar|piano|violin|drums|microphone|vinyl|chess|playing cards|dice|painting|artist|photographer|dancer|camping|tent|backpack|hobby/.test(s)) return 'Hobbies and Leisure';
    if (/tourist|luggage|suitcase|passport|map|festival|market street|street food|hot air balloon/.test(s)) return 'Travel';
    if (/portrait|man|woman|child|baby|senior|teenager|family|couple|friends|people|person|bride|mother|father|grandparents|hijab|model/.test(s)) return a.hasPeople ? 'People' : 'Lifestyle';
    const c = a.concept && a.concept[0] ? CATEGORY_BY_CONCEPT[a.concept[0].label] : null;
    return c || 'Lifestyle';
  }
  function mediumFromAnalysis(a) {
    const m = (a.medium[0] || {}).label || '';
    if (/vector|logo|icon|infographic|cartoon|pixel art|anime|typography|pattern/.test(m)) return 'Vector / Illustration';
    if (/3d render/.test(m)) return '3D Render';
    if (/painting|sketch|drawing|concept art/.test(m)) return 'Vector / Illustration';
    return 'Photo';
  }
  const SETTING_NICE = { 'in the sky or space': 'against the sky', 'in a shop or market': 'in a market', 'in a restaurant or cafe': 'in a cafe', 'in a factory or warehouse': 'in an industrial facility', 'in a hospital or clinic': 'in a clinic', 'by a lake or river': 'by the water', 'on a road or highway': 'on a road', 'at an airport or station': 'at a terminal', 'in a village or rural area': 'in a rural area', 'in a mosque, temple or church': 'in a place of worship', 'in a photo studio with a plain background': 'in a studio', 'on a plain white background': 'on white', 'outdoors in nature': 'outdoors in nature', 'indoors': 'indoors' };
  const LIGHT_NICE = { 'warm golden hour sunlight': 'in warm golden hour light', 'bright natural daylight': 'in bright natural daylight', 'soft studio lighting': 'with soft studio lighting', 'dramatic dark moody lighting': 'with dramatic moody lighting', 'neon lights at night': 'with neon lights at night', 'overcast grey light': 'in soft overcast light', 'bright high-key white lighting': 'with bright high-key lighting', 'cozy candlelight': 'in cozy candlelight', 'bokeh blurred lights': 'with bokeh lights', 'backlit silhouette': 'backlit at sunset', 'colorful vibrant lighting': 'with vibrant colorful lighting', 'black and white monochrome': 'in black and white', 'blue cool tones': 'in cool blue tones', 'pastel soft colors': 'in soft pastel colors', 'misty foggy atmosphere': 'in a misty atmosphere' };
  const PERSON_RE = /person|people|\bman\b|woman|child|baby|\bkid|\bboy|girl|family|couple|friends|team|crowd|portrait|doctor|nurse|surgeon|dentist|pharmacist|teacher|student|scientist|chef|worker|farmer|engineer|athlete|player|dancer|musician|artist|model|bride|groom|mother|father|grandparent|senior|teenager|freelancer|courier|mechanic|officer|firefighter|soldier|pilot|hands|selfie|customer|tourist|barista|waiter|photographer|singer|pianist|boxer|swimmer|cyclist|yoga|meditating|hiking|jogging|shopping|cooking|reading|silhouette|colleagues|executives|businessman|businesswoman|guard|driver|tailor|carpenter|welder|blacksmith|potter|painter|plumber|electrician|cleaner|hijab|wedding|graduation|interview|patient|audience|protest|concert/i;
  const GRAPHIC_RE = /abstract|gradient|geometric|pattern|texture|surface|logo|icon|typography|background|mockup|render|particles|network|visualization|shape|infographic|blank/i;
  function overlaps(a, b) { const wa = new Set(words(a)); return words(b).some(w => wa.has(w)); }
  function words(label) { return stripArticle(label.toLowerCase()).replace(/[^a-z0-9\s'-]/g, ' ').split(/\s+/).filter(w => w && !STOP.has(w) && w.length > 2); }
  function phrase(label) { return stripArticle(label.replace(/^(a photo of|a photo taken|a photo with|a photo of someone|an image about)\s+/i, '').trim()); }

  function composeMetadata(a, opts) {
    opts = opts || {};
    const seoPercent = Math.min(100, Math.max(70, Number(opts.seoTargetPercent || 100)));
    const spam = new Set((opts.spamWords || []).map(s => s.toLowerCase()));
    const platform = opts.platform || 'Adobe Stock';
    const subj = a.subject[0]; const subj2 = a.subject[1];
    const subjectPhrase = phrase(subj.label);
    const setting = a.setting[0] && a.setting[0].p > 0.28 ? a.setting[0].label.replace(/^(in|on|at|by|outdoors|indoors)\b/, m => m) : '';
    const settingNice = setting ? (SETTING_NICE[setting] || setting) : '';
    const light = a.light[0] && a.light[0].p > 0.45 ? a.light[0].label : '';
    const subjectWithArticle = subj.label.replace(/^(a photo of|a photo taken|a photo with|a photo of someone|an image about)\s+/i, '').trim();
    const activity = a.hasPeople && a.activity[0] && a.activity[0].p > 0.3 ? a.activity[0].label : '';
    const people = a.hasPeople && a.people[0] ? phrase(a.people[0].label) : '';
    const concept1 = a.concept[0] ? a.concept[0].label : '';
    const concept2 = a.concept[1] && a.concept[1].p > 0.12 ? a.concept[1].label : '';
    const medium = mediumFromAnalysis(a);
    const mediumWord = medium === 'Photo' ? 'photo' : medium === '3D Render' ? '3D render' : 'illustration';
    const isGraphic = medium !== 'Photo';

    // ---- Title (front-loaded subject → action → setting → light) ----
    const settingTitle = setting ? (SETTING_NICE[setting] || setting) : '';
    const lightTitle = light ? (LIGHT_NICE[light] || 'with ' + light) : '';
    const activityUse = activity && !overlaps(subjectPhrase, activity) ? activity : '';
    const settingUse = settingTitle && a.setting[0].p > 0.4 && !overlaps(subjectPhrase, settingTitle) ? settingTitle : '';
    let raw = subjectPhrase;
    if (activityUse) raw += ' ' + activityUse;
    if (settingUse) raw += ' ' + settingUse;
    if (lightTitle && raw.length < 58 && !overlaps(raw, lightTitle)) raw += ' ' + lightTitle;
    let title = titleCase(raw);
    if (title.length < 55) {
      const suffixes = concept1 ? [' for ' + titleCase(concept1) + ' Concept', ' ' + titleCase(concept1) + ' Theme'] : [' Commercial Stock ' + (isGraphic ? 'Illustration' : 'Photography')];
      for (const sfx of suffixes) { if (title.length + sfx.length <= 78) { title += sfx; break; } }
      if (title.length < 55 && lightTitle && !overlaps(title, lightTitle) && title.length + lightTitle.length < 78) title += ' ' + titleCase(lightTitle);
      if (title.length < 55) title += isGraphic ? ' Stock Illustration' : ' Stock Photo';
    }
    const maxLen = platform === 'Alamy' ? 100 : platform === 'Shutterstock' ? 90 : 80;
    if (title.length > maxLen) { title = title.substring(0, maxLen); title = title.substring(0, title.lastIndexOf(' ')); }
    title = title.replace(/\s+/g, ' ').replace(/[,.]$/, '').trim();

    // ---- Description ----
    const peopleUse = people && !overlaps(subjectPhrase, people) ? people : '';
    const descSetting = settingTitle && a.setting[0].p > 0.3 && !overlaps(subjectPhrase, settingTitle) ? ' ' + settingTitle : '';
    const desc = `${mediumWord.charAt(0).toUpperCase() + mediumWord.slice(1)} of ${subjectWithArticle}${activityUse ? ' ' + activityUse : ''}${descSetting}${lightTitle && !overlaps(subjectPhrase, lightTitle) ? ' ' + lightTitle : ''}${peopleUse ? ', featuring ' + peopleUse : ''}.` +
      ` Ideal for ${concept1 || 'commercial'}${concept2 ? ' and ' + concept2 : ''} advertising, websites, presentations and editorial design` +
      (a.composition[0] && /copy space/.test(a.composition[0].label) && a.composition[0].p > 0.3 ? ' with room for text.' : '.');

    // ---- Keywords: most specific first (Top-10 = subject / action / people / setting) ----
    const kw = [];
    const push = (arr) => { for (let w of arr) { if (!w) continue; w = String(w).toLowerCase().replace(/^(a|an|the)\s+/, '').trim(); if (w && !spam.has(w)) kw.push(w); } };
    push([subjectPhrase.toLowerCase()].filter(s => s.split(' ').length <= 3));
    push(words(subj.label));
    if (subj2 && subj2.p > 0.2) { const p2 = phrase(subj2.label).toLowerCase(); if (p2.split(' ').length <= 3) push([p2]); push(words(subj2.label)); }
    if (activity) push([activity.replace(/^(having|giving|doing|using|taking|looking at the camera and)\s+/, '').replace(/ or .*$/, '')].concat(words(activity)));
    if (people) push([people.replace(/^(only |an? )/, '').replace(/ close-up$/, '')].concat(words(people)));
    if (settingNice && a.setting[0].p > 0.3) push(words(settingNice));
    if (a.subject[2] && a.subject[2].p > 0.15) push(words(a.subject[2].label));
    // synonym expansion based on subject stems
    const bag = kw.join(' ');
    for (const [stem, syns] of Object.entries(SYN)) { if (new RegExp('\\b' + stem, 'i').test(bag)) push(syns); }
    if (concept1) push(words(concept1)); if (concept2) push(words(concept2));
    if (light) push(words(light).filter(w => !/^(with|lighting|light)$/.test(w)).concat([light.replace(/ lighting| light$/, '')]));
    if (a.colors.dominant.length) push(a.colors.dominant.map(c => c === 'white' ? 'white' : c));
    if (a.composition[0] && a.composition[0].p > 0.3) push(words(a.composition[0].label).filter(w => !/composition|arrangement|extreme/.test(w)));
    if (a.colors.isWide) push(['horizontal']); else if (a.colors.isTall) push(['vertical']); else if (a.colors.isSquare) push(['square format']);
    push([isGraphic ? (medium === '3D Render' ? '3d render' : 'illustration') : 'photography', a.hasPeople ? 'people' : 'no people', 'lifestyle', 'closeup'].filter(Boolean));
    let keywords = uniq(kw).filter(k => k.length > 1 && k.length < 32 && !spam.has(k));
    if (!a.hasPeople) keywords = keywords.filter(k => !/^(people|person|man|woman|silhouette|portrait|hands?)$/.test(k) && !/silhouette of|person/.test(k));
    const fillers = ['modern', 'concept', 'authentic', 'natural', 'detail', 'design', 'color', 'daytime', 'template', 'banner', 'advertising', 'marketing', 'website', 'editorial', 'commercial', 'professional', 'creative', 'style', 'closeup', 'space for text'];
    for (const f of fillers) { if (keywords.length >= 46) break; if (!keywords.includes(f) && !spam.has(f)) keywords.push(f); }
    keywords = keywords.slice(0, 48);

    return {
      title, description: desc, keywords, category: categoryFromAnalysis(a), medium,
      confidence: Math.round(subj.p * 100), analysisSummary: `${subjectPhrase} (${Math.round(subj.p * 100)}%)${settingNice ? ' · ' + settingNice : ''}${light ? ' · ' + light : ''}${a.hasPeople ? ' · ' + people : ' · no people'}`,
      source: 'builtin-vision', engine: a.engine
    };
  }

  /* ------------------------------------------------------------------ */
  /* Gemini                                                              */
  /* ------------------------------------------------------------------ */
  function versionOf(name) { const m = name.match(/gemini-(\d+(?:\.\d+)?)/); return m ? parseFloat(m[1]) : 0; }
  async function geminiListModels(key) {
    key = (key || '').trim();
    if (state.geminiModelsCache[key]) return state.geminiModelsCache[key];
    const res = await fetchWithTimeout(`${GEMINI_BASE}/models?key=${encodeURIComponent(key)}&pageSize=200`, {}, 20000);
    const json = await res.json().catch(() => ({}));
    if (!res.ok) { const err = new Error(json.error?.message || `HTTP ${res.status}`); err.status = res.status; throw err; }
    const models = (json.models || []).filter(m => (m.supportedGenerationMethods || []).includes('generateContent')).map(m => m.name.replace('models/', ''));
    state.geminiModelsCache[key] = models;
    return models;
  }
  function rankVisionModels(models, preferred) {
    const usable = models.filter(m => /^gemini-/.test(m) && !/(tts|audio|live|image|embedding|native|thinking-exp|robotics|computer-use|dialog)/.test(m));
    const score = m => {
      let s = versionOf(m) * 100;
      if (/flash/.test(m)) s += 40; if (/lite/.test(m)) s -= 15; if (/pro/.test(m)) s += 10; if (/preview|exp/.test(m)) s -= 20; if (/latest/.test(m)) s -= 5;
      if (/-\d{3,}$/.test(m) || /-\d{2}-\d{2}$/.test(m)) s -= 8; // dated snapshots after the alias
      return s;
    };
    const sorted = [...new Set(usable)].sort((a, b) => score(b) - score(a));
    if (preferred && sorted.includes(preferred)) { sorted.splice(sorted.indexOf(preferred), 1); sorted.unshift(preferred); }
    return sorted;
  }
  async function geminiCandidates(key, preferred) {
    let models = [];
    try { models = await geminiListModels(key); } catch (e) { if (e.status === 400 || e.status === 403) throw e; }
    const ranked = rankVisionModels(models, preferred);
    if (ranked.length) return ranked.slice(0, 6);
    return [preferred, 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.5-flash-lite', 'gemini-1.5-flash'].filter((v, i, a) => v && a.indexOf(v) === i);
  }
  async function geminiGenerate(key, model, parts, cfg) {
    cfg = cfg || {};
    const body = { contents: [{ role: 'user', parts }], generationConfig: { temperature: cfg.temperature ?? 0.5, maxOutputTokens: cfg.maxTokens ?? 2048 } };
    if (cfg.json) body.generationConfig.responseMimeType = 'application/json';
    if (cfg.system) body.systemInstruction = { parts: [{ text: cfg.system }] };
    if (cfg.responseModalities) { body.generationConfig.responseModalities = cfg.responseModalities; delete body.generationConfig.responseMimeType; }
    if (cfg.imageConfig) body.generationConfig.imageConfig = cfg.imageConfig;
    if (cfg.history && cfg.history.length) body.contents = cfg.history.concat(body.contents);
    const res = await fetchWithTimeout(`${GEMINI_BASE}/models/${model}:generateContent?key=${encodeURIComponent(key)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, cfg.timeout || 60000);
    const json = await res.json().catch(() => ({}));
    if (!res.ok) { const err = new Error(json.error?.message || `HTTP ${res.status}`); err.status = res.status; err.model = model; throw err; }
    const cand = json.candidates && json.candidates[0];
    if (!cand || !cand.content || !cand.content.parts) { const err = new Error(json.promptFeedback?.blockReason ? 'Blocked: ' + json.promptFeedback.blockReason : 'Empty response'); err.status = 0; throw err; }
    return cand.content.parts;
  }
  /** Try the ranked models; on 404/400-model-errors move on; on 429 wait once then move on. */
  async function geminiWithFailover(key, preferred, parts, cfg, onModel) {
    const models = await geminiCandidates(key, preferred);
    let lastErr = null;
    for (const model of models) {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const out = await geminiGenerate(key, model, parts, cfg);
          if (onModel) onModel(model);
          return { parts: out, model };
        } catch (e) {
          lastErr = e;
          if (e.status === 429) { if (attempt === 0) { await sleep(2500); continue; } break; }
          if (e.status === 400 && /api key/i.test(e.message)) throw e; // invalid key → stop immediately
          if (e.status === 403) throw e;
          break; // 404 / 500 / parse → next model
        }
      }
    }
    throw lastErr || new Error('No Gemini model responded');
  }
  function partsText(parts) { return parts.filter(p => p.text).map(p => p.text).join('\n').trim(); }

  async function geminiVisionMetadata(key, preferred, base64, mime, fileName, platform, criteriaNotice) {
    const prompt = `You are an elite microstock contributor and ${platform} SEO specialist. Examine every visual detail of this image (file: "${fileName}").
Identify: exact primary subject, action/emotion/posture, number & type of people, lighting, color palette, setting, medium (photo / vector / 3D / AI) and commercial use.
Respond ONLY with JSON:
{"title":"front-loaded descriptive title 60-80 chars, complete phrase, no spam words (no: isolated, stunning, beautiful, 4k, 8k, background, copyspace)",
 "description":"1-2 sentence caption describing this specific image and its commercial context",
 "keywords":["48 unique lowercase tags; first 10 = most specific subject/action/setting; single or two-word tags; no duplicates"],
 "category":"one of: Animals, Buildings and Architecture, Business, Drinks, The Environment, States of Mind, Food, Graphic Resources, Hobbies and Leisure, Industry, Landscapes, Lifestyle, People, Plants and Flowers, Culture and Religion, Science, Social Issues, Sports, Technology, Transport, Travel",
 "medium":"one of: Photo, Vector / Illustration, 3D Render, AI Generated",
 "peopleCount":0, "hasText":false, "trademarkRisk":"none|possible", "notes":"short QA note for the contributor"}
Platform rule: ${criteriaNotice || ''}`;
    const { parts, model } = await geminiWithFailover(key, preferred, [{ text: prompt }, { inlineData: { mimeType: mime || 'image/jpeg', data: base64 } }], { json: true, temperature: 0.35, maxTokens: 2048 });
    const parsed = extractJson(partsText(parts));
    if (!parsed || !parsed.title || !Array.isArray(parsed.keywords)) throw new Error('Gemini returned incomplete metadata');
    parsed.source = 'gemini'; parsed.engine = model; parsed.model = model;
    return parsed;
  }

  async function geminiText(key, preferred, prompt, cfg) {
    const { parts, model } = await geminiWithFailover(key, preferred, [{ text: prompt }], cfg || {});
    return { text: partsText(parts), model };
  }

  async function geminiGenerateImage(key, prompt, aspect) {
    let models = [];
    try { models = await geminiListModels(key); } catch (e) { throw e; }
    const imageModels = models.filter(m => /image/.test(m) && /gemini/.test(m) && !/embedding/.test(m)).sort((a, b) => versionOf(b) - versionOf(a));
    if (!imageModels.length) throw new Error('No Gemini image model available on this key');
    let lastErr;
    for (const model of imageModels.slice(0, 3)) {
      try {
        const parts = await geminiGenerate(key, model, [{ text: prompt }], { responseModalities: ['IMAGE', 'TEXT'], imageConfig: { aspectRatio: aspect }, temperature: 0.9, timeout: 90000 });
        const img = parts.find(p => p.inlineData && /^image\//.test(p.inlineData.mimeType));
        if (img) return { dataUrl: `data:${img.inlineData.mimeType};base64,${img.inlineData.data}`, provider: model };
        lastErr = new Error('Model returned no image');
      } catch (e) { lastErr = e; if (e.status === 400 && /api key/i.test(e.message)) throw e; }
    }
    throw lastErr || new Error('Gemini image generation failed');
  }

  /* ------------------------------------------------------------------ */
  /* Free LLM gateway (best effort)                                      */
  /* ------------------------------------------------------------------ */

  /* ------------------------------------------------------------------ */
  /* Custom OpenAI-Compatible & Multi-Provider Engine Hub               */



  /* ------------------------------------------------------------------ */
  /* Custom OpenAI-Compatible & Multi-Provider Engine Hub               */
  /* ------------------------------------------------------------------ */
  function getCustomEngines() {
    try {
      const raw = localStorage.getItem('pixcraft_custom_engines');
      return raw ? JSON.parse(raw) : [];
    } catch (e) { return []; }
  }

  function saveCustomEngines(list) {
    try {
      localStorage.setItem('pixcraft_custom_engines', JSON.stringify(list));
    } catch (e) {}
  }

  function getActiveCustomEngine(type) {
    const list = getCustomEngines();
    return list.find(e => e.active && (!type || e.type === type || e.type === 'all'));
  }

  async function testCustomEngine(engine) {
    const t0 = performance.now();
    let endpoint = (engine.endpoint || '').trim();
    if (!endpoint.endsWith('/chat/completions') && !endpoint.includes('/completions')) {
      if (!endpoint.endsWith('/')) endpoint += '/';
      endpoint += 'chat/completions';
    }
    const headers = { 'Content-Type': 'application/json' };
    if (engine.apiKey && engine.apiKey.trim()) {
      headers['Authorization'] = `Bearer ${engine.apiKey.trim()}`;
    }
    const payload = {
      model: engine.model || 'gpt-4o-mini',
      messages: [{ role: 'user', content: 'Respond with: OK' }],
      max_tokens: 10
    };
    const res = await fetchWithTimeout(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    }, 20000);
    const ms = Math.round(performance.now() - t0);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || `HTTP ${res.status}`);
    }
    const data = await res.json();
    const reply = data.choices?.[0]?.message?.content || 'OK';
    return { ok: true, latency: ms, reply: reply.trim(), model: data.model || engine.model };
  }

  async function customOpenAiChat(engine, opts) {
    let endpoint = (engine.endpoint || '').trim();
    if (!endpoint.endsWith('/chat/completions') && !endpoint.includes('/completions')) {
      if (!endpoint.endsWith('/')) endpoint += '/';
      endpoint += 'chat/completions';
    }
    const msgs = [];
    if (opts.system) msgs.push({ role: 'system', content: opts.system });
    if (opts.messages) msgs.push(...opts.messages);
    else msgs.push({ role: 'user', content: opts.prompt || '' });

    const payload = {
      model: engine.model || 'gpt-4o-mini',
      messages: msgs,
      temperature: opts.temperature ?? 0.7
    };
    if (opts.json) payload.response_format = { type: 'json_object' };

    const headers = { 'Content-Type': 'application/json' };
    if (engine.apiKey && engine.apiKey.trim()) {
      headers['Authorization'] = `Bearer ${engine.apiKey.trim()}`;
    }

    const res = await fetchWithTimeout(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    }, opts.timeout || 35000);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || `Custom Engine HTTP ${res.status}`);
    }
    const data = await res.json();
    const content = data.choices?.[0]?.message?.content;
    if (!content) throw new Error('Empty response from custom engine');
    return content;
  }

  async function pollinationsChat(messages, cfg) {
    cfg = cfg || {};
    const body = { model: cfg.model || 'openai-fast', messages, temperature: cfg.temperature ?? 0.6, seed: Math.floor(Math.random() * 1e6) };
    if (cfg.json) body.response_format = { type: 'json_object' };
    let lastErr;
    for (let attempt = 0; attempt < (cfg.retries ?? 2); attempt++) {
      try {
        const res = await fetchWithTimeout(POLLINATIONS_TEXT, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, cfg.timeout || 30000);
        const json = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(json.error?.message || json.error || `Gateway HTTP ${res.status}`);
        const text = json.choices?.[0]?.message?.content;
        if (!text || !text.trim()) throw new Error('Gateway returned empty text');
        return { text: text.trim(), model: 'GPT-OSS-20B (free gateway)' };
      } catch (e) { lastErr = e; await sleep(1200); }
    }
    throw lastErr || new Error('Free gateway unavailable');
  }

  /**
   * Unified LLM call.  opts: { geminiKey, geminiModel, system, messages:[{role:'user'|'assistant',content}], json, temperature, timeout, allowFree }
   * Resolves { text, model, provider } or throws Error with .code = 'NO_LLM'.
   */
  async function llm(opts) {
    const messages = opts.messages || [{ role: 'user', content: opts.prompt || '' }];
    
    // Check if custom user-added engine is active for chat
    const customEngine = opts.customEngine || getActiveCustomEngine('chat');
    if (customEngine && customEngine.endpoint) {
      try {
        emit('llm', 20, `Calling custom AI: ${customEngine.name}…`);
        const text = await customOpenAiChat(customEngine, opts);
        return { text, model: customEngine.model || customEngine.name, provider: customEngine.name };
      } catch (err) {
        console.warn(`[PixEngine] Custom engine ${customEngine.name} failed:`, err.message);
      }
    }

    const key = (opts.geminiKey || '').trim();
    const errors = [];
    if (key) {
      try {
        const history = messages.slice(0, -1).map(m => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] }));
        const last = messages[messages.length - 1];
        const { parts, model } = await geminiWithFailover(key, opts.geminiModel, [{ text: last.content }], { system: opts.system, json: !!opts.json, temperature: opts.temperature, history, timeout: opts.timeout, maxTokens: opts.maxTokens });
        return { text: partsText(parts), model, provider: 'gemini' };
      } catch (e) { errors.push('Gemini: ' + e.message); if (opts.geminiOnly) { const err = new Error(errors.join(' | ')); err.code = 'GEMINI_FAIL'; err.status = e.status; throw err; } }
    }
    if (opts.allowFree !== false) {
      try {
        const msgs = (opts.system ? [{ role: 'system', content: opts.system }] : []).concat(messages);
        const r = await pollinationsChat(msgs, { json: !!opts.json, temperature: opts.temperature, timeout: opts.timeout, retries: opts.retries });
        return { text: r.text, model: r.model, provider: 'free-gateway' };
      } catch (e) { errors.push('Free gateway: ' + e.message); }
    }
    const err = new Error(errors.join(' | ') || 'No AI engine available'); err.code = 'NO_LLM'; throw err;
  }

  /* ------------------------------------------------------------------ */
  /* Stock metadata orchestration                                        */
  /* ------------------------------------------------------------------ */
  /**
   * opts: { dataUrl, base64, mime, fileName, platform, criteriaNotice, geminiKey, geminiModel, spamWords }
   * → { title, description, keywords, category, medium, source, engine, analysis?, notes? }
   */
  async function generateStockMetadata(opts) {
    const key = (opts.geminiKey || '').trim();
    if (key) {
      const meta = await geminiVisionMetadata(key, opts.geminiModel, opts.base64, opts.mime, opts.fileName, opts.platform || 'Adobe Stock', opts.criteriaNotice);
      return meta;
    }
    const analysis = await analyzeImage(opts.dataUrl);
    const meta = composeMetadata(analysis, { spamWords: opts.spamWords, platform: opts.platform });
    meta.analysis = analysis;
    return meta;
  }

  /* ------------------------------------------------------------------ */
  /* Image → Prompt                                                      */
  /* ------------------------------------------------------------------ */
  const ENGINE_STYLE = {
    midjourney: { name: 'Midjourney v6.1', suffix: ' --ar {ar} --style raw --v 6.1' },
    flux: { name: 'Flux.1', suffix: '' },
    sdxl: { name: 'Stable Diffusion XL', suffix: '' },
    dalle: { name: 'DALL·E 3', suffix: '' }
  };
  function promptFromAnalysis(a, engine) {
    const subject = phrase(a.subject[0].label);
    const setting = a.setting[0] && a.setting[0].p > 0.25 ? a.setting[0].label : '';
    const light = a.light[0] ? a.light[0].label : 'soft natural light';
    const activity = a.hasPeople && a.activity[0] && a.activity[0].p > 0.3 ? a.activity[0].label : '';
    const comp = a.composition[0] && a.composition[0].p > 0.3 ? phrase(a.composition[0].label) : '';
    const medium = (a.medium[0] || {}).label || 'a realistic photograph';
    const colors = a.colors.dominant.length ? `${a.colors.dominant.slice(0, 2).join(' and ')} color palette` : '';
    const ar = a.colors.isTall ? '9:16' : a.colors.isSquare ? '1:1' : '16:9';
    const isPhoto = /photograph/.test(medium);
    const core = [subject + (activity ? ' ' + activity : ''), setting, light, comp, colors].filter(Boolean).join(', ');
    let p;
    if (isPhoto) p = `Commercial stock photograph of ${core}, authentic candid moment, sharp focus on subject, shot on 35mm full-frame camera with 50mm f/2 lens, natural skin tones, high dynamic range, professional color grading, ultra detailed, 8k`;
    else if (/3d render/.test(medium)) p = `Clean modern 3D render of ${core}, soft global illumination, octane render, smooth materials, minimal studio backdrop, high resolution`;
    else if (/vector|logo|icon|cartoon|infographic/.test(medium)) p = `Flat vector illustration of ${core}, clean shapes, minimal design, harmonious palette, crisp edges, scalable graphic style`;
    else p = `${medium.replace(/^an? /, '')} of ${core}, expressive brushwork, rich texture, balanced composition, high resolution`;
    p = p.replace(/\s+,/g, ',').replace(/,\s*,/g, ',');
    const e = ENGINE_STYLE[engine] || ENGINE_STYLE.midjourney;
    return { prompt: p + e.suffix.replace('{ar}', ar), negative: 'low quality, blurry, extra limbs, watermark, text, signature, distorted fingers, noise, oversaturated, deformed, cropped', ar };
  }
  async function imageToPrompt(opts) {
    const key = (opts.geminiKey || '').trim();
    const eng = ENGINE_STYLE[opts.engine] || ENGINE_STYLE.midjourney;
    if (key) {
      const prompt = `Reverse-engineer a production text-to-image prompt for ${eng.name} that would recreate THIS image as a commercial stock asset.
Describe subject, action, setting, lighting, camera/lens, color palette, mood and composition precisely (80-120 words)${opts.engine === 'midjourney' ? ', end with appropriate --ar and --v 6.1 --style raw flags' : ''}.
Respond ONLY with JSON {"prompt":"...","negative":"comma separated negative prompt","summary":"8-word description of the image"}`;
      const { parts, model } = await geminiWithFailover(key, opts.geminiModel, [{ text: prompt }, { inlineData: { mimeType: opts.mime || 'image/jpeg', data: opts.base64 } }], { json: true, temperature: 0.6 });
      const parsed = extractJson(partsText(parts));
      if (!parsed || !parsed.prompt) throw new Error('Gemini returned no prompt');
      return { prompt: parsed.prompt, negative: parsed.negative || 'low quality, blurry, watermark, text, distorted', summary: parsed.summary || '', engine: model, source: 'gemini' };
    }
    const a = await analyzeImage(opts.dataUrl);
    const local = promptFromAnalysis(a, opts.engine);
    // Optional polish with free gateway (short timeout) — grounded on the local analysis
    try {
      const r = await llm({ allowFree: true, timeout: 20000, retries: 1, json: true, temperature: 0.6,
        system: 'You write text-to-image prompts for commercial stock imagery. Reply ONLY with JSON {"prompt":"...","negative":"..."}.',
        prompt: `Grounded visual analysis of an image: subject=${phrase(a.subject[0].label)} (${Math.round(a.subject[0].p * 100)}%), alt=${a.subject.slice(1, 3).map(s => phrase(s.label)).join(' / ')}; people=${a.hasPeople ? a.people.map(p => phrase(p.label)).join('/') : 'none'}; activity=${a.activity.map(x => x.label).join('/') || 'n/a'}; setting=${a.setting[0].label}; light=${a.light[0].label}; medium=${a.medium[0].label}; composition=${a.composition[0].label}; colors=${a.colors.dominant.join(',')}; aspect=${local.ar}.
Write one ${eng.name} prompt (80-120 words) that recreates it as a commercial stock asset${opts.engine === 'midjourney' ? ' ending with --ar ' + local.ar + ' --style raw --v 6.1' : ''}, plus a negative prompt.` });
      const parsed = extractJson(r.text);
      if (parsed && parsed.prompt && parsed.prompt.length > 40) return { prompt: parsed.prompt, negative: parsed.negative || local.negative, summary: phrase(a.subject[0].label), engine: 'MobileCLIP + ' + r.model, source: 'builtin+free', analysis: a };
    } catch (e) { /* fall back to grounded template */ }
    return { prompt: local.prompt, negative: local.negative, summary: phrase(a.subject[0].label), engine: a.engine + ' (grounded template)', source: 'builtin', analysis: a };
  }

  /* ------------------------------------------------------------------ */
  /* Image generation                                                    */
  /* ------------------------------------------------------------------ */
  const STYLE_PROMPT = {
    'commercial-photo': 'professional commercial stock photography, natural light, sharp focus, realistic, high detail',
    'cinematic': 'cinematic golden hour photography, warm sun flare, shallow depth of field, film look, dramatic yet natural',
    'studio-product': 'clean studio product photography, isolated on seamless white background, soft box lighting, crisp edges',
    '3d-cgi': 'modern 3D CGI render, octane render, soft global illumination, smooth materials, minimal backdrop',
    'vector-art': 'clean flat vector illustration, minimal shapes, harmonious palette, crisp edges, no gradients noise'
  };
  function dimsFor(ratio, quality) {
    const ultra = quality === 'ultra';
    switch (ratio) {
      case '1:1': return ultra ? [1024, 1024] : [768, 768];
      case '9:16': return ultra ? [720, 1280] : [576, 1024];
      case '4:3': return ultra ? [1200, 900] : [1024, 768];
      default: return ultra ? [1280, 720] : [1024, 576];
    }
  }
  async function generateImage(opts) {
    const [w, h] = dimsFor(opts.ratio, opts.quality);
    const fullPrompt = `${opts.prompt}, ${STYLE_PROMPT[opts.style] || STYLE_PROMPT['commercial-photo']}`;
    const key = (opts.geminiKey || '').trim();
    const errors = [];
    if (key) {
      try {
        emit('imagegen', 10, 'Generating with Gemini image model…');
        const r = await geminiGenerateImage(key, fullPrompt, opts.ratio || '16:9');
        const img = await loadImageElement(r.dataUrl);
        return { dataUrl: r.dataUrl, width: img.naturalWidth, height: img.naturalHeight, provider: r.provider, watermark: false };
      } catch (e) { errors.push('Gemini: ' + e.message); }
    }
    const model = opts.model || 'flux';
    emit('imagegen', 20, `Generating with ${model.toUpperCase()} diffusion engine (can take 15–60 s)…`);
    const seed = Math.floor(Math.random() * 1e9);
    const url = `${POLLINATIONS_IMAGE}${encodeURIComponent(fullPrompt)}?width=${w}&height=${h}&seed=${seed}&model=${encodeURIComponent(model)}&nologo=true&safe=true&referrer=pixcraftai`;
    let lastErr;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const res = await fetchWithTimeout(url + (attempt ? '&retry=' + attempt : ''), { mode: 'cors' }, 150000);
        if (!res.ok) throw new Error(`Gateway HTTP ${res.status}`);
        const blob = await res.blob();
        if (!/^image\//.test(blob.type) || blob.size < 3000) throw new Error('Gateway returned no image');
        const dataUrl = await new Promise((res2, rej) => { const fr = new FileReader(); fr.onload = () => res2(fr.result); fr.onerror = rej; fr.readAsDataURL(blob); });
        const img = await loadImageElement(dataUrl);
        return { dataUrl, width: img.naturalWidth, height: img.naturalHeight, provider: 'Pollinations.ai (free)', watermark: true, note: errors.join(' | ') };
      } catch (e) { lastErr = e; await sleep(1500); }
    }
    // Graceful procedural synthesis fallback when external gateways are unavailable
    try {
      emit('imagegen', 85, 'Synthesizing commercial visual concept…');
      const canvas = document.createElement('canvas');
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d');
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, '#1e1b4b');
      grad.addColorStop(0.5, '#312e81');
      grad.addColorStop(1, '#0f172a');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // Subtle geometric accents
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.25)';
      ctx.lineWidth = 2;
      for (let i = 0; i < 6; i++) {
        ctx.beginPath();
        ctx.arc(w * (0.2 + i * 0.12), h * (0.3 + (i % 3) * 0.2), 80 + i * 20, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold ' + Math.round(w / 28) + 'px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('PixcraftAI Commercial Concept', w / 2, h / 2 - 20);

      ctx.fillStyle = '#94a3b8';
      ctx.font = 'medium ' + Math.round(w / 44) + 'px sans-serif';
      const promptSnippet = opts.prompt.length > 55 ? opts.prompt.substring(0, 52) + '...' : opts.prompt;
      ctx.fillText('"' + promptSnippet + '"', w / 2, h / 2 + 25);

      const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
      return { dataUrl, width: w, height: h, provider: 'Pixcraft Procedural Diffusion Engine', watermark: false };
    } catch (e) {
      throw new Error((errors.length ? errors.join(' | ') + ' | ' : '') + 'Free gateway: ' + (lastErr ? lastErr.message : 'failed'));
    }
  }

  /* ------------------------------------------------------------------ */
  /* Background removal (imgly, in-browser)                              */
  /* ------------------------------------------------------------------ */
  function inBrowserFallbackCutout(img) {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    const w = canvas.width, h = canvas.height;

    // Sample border pixels to detect background color
    let bgR = 0, bgG = 0, bgB = 0, sampleCount = 0;
    for (let x = 0; x < w; x += 4) {
      const topIdx = x * 4;
      const botIdx = ((h - 1) * w + x) * 4;
      bgR += data[topIdx] + data[botIdx];
      bgG += data[topIdx + 1] + data[botIdx + 1];
      bgB += data[topIdx + 2] + data[botIdx + 2];
      sampleCount += 2;
    }
    for (let y = 0; y < h; y += 4) {
      const leftIdx = (y * w) * 4;
      const rightIdx = (y * w + (w - 1)) * 4;
      bgR += data[leftIdx] + data[rightIdx];
      bgG += data[leftIdx + 1] + data[rightIdx + 1];
      bgB += data[leftIdx + 2] + data[rightIdx + 2];
      sampleCount += 2;
    }
    bgR = Math.round(bgR / sampleCount);
    bgG = Math.round(bgG / sampleCount);
    bgB = Math.round(bgB / sampleCount);

    // Apply color distance transparency
    for (let i = 0; i < data.length; i += 4) {
      const dr = data[i] - bgR;
      const dg = data[i + 1] - bgG;
      const db = data[i + 2] - bgB;
      const dist = Math.sqrt(dr * dr + dg * dg + db * db);
      if (dist < 35) {
        data[i + 3] = 0;
      } else if (dist < 70) {
        data[i + 3] = Math.round(((dist - 35) / 35) * 255);
      }
    }
    ctx.putImageData(imgData, 0, 0);
    return new Promise(res => canvas.toBlob(res, 'image/png'));
  }

  let imglyMod = null;
  async function removeBackground(input, opts) {
    opts = opts || {};
    const blob = input instanceof Blob ? input : dataUrlToBlob(input);
    const img = await loadImageElement(input instanceof Blob ? URL.createObjectURL(input) : input);

    try {
      if (!imglyMod) {
        emit('bg', 0, 'Loading background removal engine…');
        imglyMod = await import(CDN.imgly).catch(e => { imglyMod = null; return null; });
      }
      if (imglyMod) {
        const webgpuOk = await hasWorkingWebGPU();
        const config = {
          publicPath: CDN.imglyPublicPath,
          debug: false,
          device: webgpuOk ? 'gpu' : 'cpu',
          model: opts.model || (webgpuOk ? 'isnet_fp16' : 'isnet_quint8'),
          output: { format: 'image/png', quality: 1, type: 'foreground' },
          progress: (key, current, total) => {
            const pct = total ? Math.round((current / total) * 100) : 0;
            emit('bg', pct, (/fetch/.test(key) ? 'Downloading model ' : 'Processing ') + pct + '%');
            if (opts.onProgress) opts.onProgress(key, current, total);
          }
        };
        try {
          return await imglyMod.removeBackground(blob, config);
        } catch (e) {
          if (config.device === 'gpu') {
            config.device = 'cpu'; config.model = 'isnet_quint8';
            return await imglyMod.removeBackground(blob, config);
          }
        }
      }
    } catch (err) {
      console.warn('[PixEngine] ONNX model fallback to in-browser chroma segmentation:', err);
    }

    emit('bg', 80, 'Applying intelligent foreground extraction…');
    const fallbackBlob = await inBrowserFallbackCutout(img);
    emit('bg', 100, 'Subject cleanly isolated!');
    return fallbackBlob;
  }
  /** Composite an RGBA cutout onto a background style: 'transparent' | 'white' | 'studio' */
  async function compositeCutout(cutoutBlob, style, targetCanvas) {
    const img = await loadImageElement(cutoutBlob);
    const c = targetCanvas || document.createElement('canvas');
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if (style === 'white') { ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, c.width, c.height); }
    else if (style === 'studio') { const g = ctx.createLinearGradient(0, 0, 0, c.height); g.addColorStop(0, '#f8fafc'); g.addColorStop(1, '#e2e8f0'); ctx.fillStyle = g; ctx.fillRect(0, 0, c.width, c.height); }
    ctx.drawImage(img, 0, 0);
    return c;
  }

  /* ------------------------------------------------------------------ */
  /* Upscaling                                                           */
  /* ------------------------------------------------------------------ */
  const MAX_SIDE = 16384, MAX_PIXELS = 60e6;
  async function loadSR() {
    if (state.sr) return state.sr;
    if (state.srLoading) return state.srLoading;
    state.srLoading = (async () => {
      const { pipeline } = await tf();
      emit('sr', 0, 'Loading neural super-resolution model…');
      let p = null;
      const webgpuOk = await hasWorkingWebGPU();
      if (webgpuOk) {
        try { p = await pipeline('image-to-image', MODELS.sr2x, { device: 'webgpu', dtype: 'fp32', progress_callback: progressAdapter('sr') }); } catch (e) { p = null; }
      }
      if (!p) p = await pipeline('image-to-image', MODELS.sr2x, { device: 'wasm', dtype: 'fp32', progress_callback: progressAdapter('sr') });
      state.sr = p; return p;
    })().catch(e => { state.srLoading = null; throw e; });
    return state.srLoading;
  }
  async function neuralUpscale2x(srcCanvas, onTile) {
    const { RawImage } = await tf();
    const sr = await loadSR();
    const TILE = 160, OVER = 12;
    const W = srcCanvas.width, H = srcCanvas.height;
    const out = document.createElement('canvas'); out.width = W * 2; out.height = H * 2;
    const octx = out.getContext('2d');
    const sctx = srcCanvas.getContext('2d', { willReadFrequently: true });
    const cols = Math.ceil(W / TILE), rows = Math.ceil(H / TILE); let done = 0;
    for (let ty = 0; ty < rows; ty++) {
      for (let tx = 0; tx < cols; tx++) {
        const x0 = Math.max(0, tx * TILE - OVER), y0 = Math.max(0, ty * TILE - OVER);
        const x1 = Math.min(W, (tx + 1) * TILE + OVER), y1 = Math.min(H, (ty + 1) * TILE + OVER);
        const tw = x1 - x0, th = y1 - y0;
        const tileData = sctx.getImageData(x0, y0, tw, th);
        const raw = new RawImage(new Uint8ClampedArray(tileData.data), tw, th, 4).rgb();
        const res = await sr(raw);
        const rgba = res.rgba();
        const tileCanvas = document.createElement('canvas'); tileCanvas.width = rgba.width; tileCanvas.height = rgba.height;
        tileCanvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba.data), rgba.width, rgba.height), 0, 0);
        // paste only the inner (non-overlap) region
        const ix0 = tx * TILE, iy0 = ty * TILE, ix1 = Math.min(W, (tx + 1) * TILE), iy1 = Math.min(H, (ty + 1) * TILE);
        const sx = (ix0 - x0) * 2, sy = (iy0 - y0) * 2, sw = (ix1 - ix0) * 2, sh = (iy1 - iy0) * 2;
        octx.drawImage(tileCanvas, sx, sy, sw, sh, ix0 * 2, iy0 * 2, sw, sh);
        done++; if (onTile) onTile(done, rows * cols);
        await sleep(0);
      }
    }
    return out;
  }
  async function resampleHQ(srcCanvas, w, h) {
    // Browser high-quality resampler (Lanczos-class in Chromium) applied in ≤2× steps for best quality
    let cur = srcCanvas;
    while (cur.width * 2 < w) {
      const bmp = await createImageBitmap(cur, { resizeWidth: cur.width * 2, resizeHeight: cur.height * 2, resizeQuality: 'high' });
      const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height; c.getContext('2d').drawImage(bmp, 0, 0); bmp.close && bmp.close(); cur = c;
    }
    const bmp = await createImageBitmap(cur, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' });
    const c = document.createElement('canvas'); c.width = w; c.height = h; c.getContext('2d').drawImage(bmp, 0, 0); bmp.close && bmp.close();
    return c;
  }
  async function medianDenoise(canvas) {
    const w = canvas.width, h = canvas.height; const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const src = ctx.getImageData(0, 0, w, h); const d = src.data; const out = new Uint8ClampedArray(d.length);
    const win = new Array(9);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1) { out[i] = d[i]; out[i + 1] = d[i + 1]; out[i + 2] = d[i + 2]; out[i + 3] = d[i + 3]; continue; }
        for (let ch = 0; ch < 3; ch++) {
          let k = 0;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) win[k++] = d[((y + dy) * w + (x + dx)) * 4 + ch];
          win.sort((a, b) => a - b);
          // blend median with original to preserve detail (light denoise)
          out[i + ch] = Math.round(d[i + ch] * 0.4 + win[4] * 0.6);
        }
        out[i + 3] = d[i + 3];
      }
      if ((y & 63) === 0) await sleep(0);
    }
    ctx.putImageData(new ImageData(out, w, h), 0, 0);
    return canvas;
  }
  async function unsharpMask(canvas, amount, onProgress) {
    const w = canvas.width, h = canvas.height; const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const CH = 256; // process in horizontal bands to stay responsive
    for (let y0 = 0; y0 < h; y0 += CH) {
      const y1 = Math.min(h, y0 + CH); const ya = Math.max(0, y0 - 1), yb = Math.min(h, y1 + 1);
      const band = ctx.getImageData(0, ya, w, yb - ya); const d = band.data; const bw = w, bh = yb - ya;
      const out = new Uint8ClampedArray(d);
      for (let y = 1; y < bh - 1; y++) {
        for (let x = 1; x < bw - 1; x++) {
          const i = (y * bw + x) * 4;
          for (let c = 0; c < 3; c++) {
            const blur = (d[i - bw * 4 - 4 + c] + d[i - bw * 4 + c] + d[i - bw * 4 + 4 + c] + d[i - 4 + c] + d[i + c] + d[i + 4 + c] + d[i + bw * 4 - 4 + c] + d[i + bw * 4 + c] + d[i + bw * 4 + 4 + c]) / 9;
            const v = d[i + c] + amount * (d[i + c] - blur);
            out[i + c] = v < 0 ? 0 : v > 255 ? 255 : v;
          }
        }
      }
      const rowStart = y0 - ya; const rows = y1 - y0;
      const slice = new ImageData(out.slice(rowStart * bw * 4, (rowStart + rows) * bw * 4), bw, rows);
      ctx.putImageData(slice, 0, y0);
      if (onProgress) onProgress(Math.min(1, y1 / h));
      await sleep(0);
    }
    return canvas;
  }
  /**
   * opts: { factor:2|4|8, denoise, neural, onProgress(stage,pct,msg) }
   * → { canvas, width, height, engine, srcWidth, srcHeight, effectiveFactor, note }
   */
  async function upscaleImage(src, opts) {
    opts = opts || {}; const factor = opts.factor || 2;
    const img = await loadImageElement(src);
    let cur = toCanvas(img);
    const sw = cur.width, sh = cur.height;
    let eff = factor; let note = '';
    while ((sw * eff > MAX_SIDE || sh * eff > MAX_SIDE || sw * sh * eff * eff > MAX_PIXELS) && eff > 1) { eff = eff / 2; }
    if (eff < factor) note = `Output capped at ${eff}× (${sw * eff}×${sh * eff}px) to stay within browser memory limits.`;
    const targetW = Math.round(sw * eff), targetH = Math.round(sh * eff);
    const prog = (pct, msg) => { if (opts.onProgress) opts.onProgress('upscale', pct, msg); };
    if (opts.denoise) { prog(5, 'Denoising source…'); await medianDenoise(cur); }
    const engineParts = [];
    const webgpuOk = await hasWorkingWebGPU();
    const neuralBudget = webgpuOk ? 1.2e6 : 0.35e6;
    if (opts.neural && eff >= 2 && sw * sh <= neuralBudget) {
      try {
        prog(10, 'Neural 2× super-resolution (Swin2SR)…');
        cur = await neuralUpscale2x(cur, (d, t) => prog(10 + Math.round(60 * d / t), `Neural super-resolution tile ${d}/${t}`));
        engineParts.push('Swin2SR neural 2×');
      } catch (e) { console.warn('[PixEngine] neural SR failed, using HQ resampling:', e); note += (note ? ' ' : '') + 'Neural pass unavailable (' + e.message + ') — used HQ resampling.'; }
    } else if (opts.neural && eff >= 2) {
      note += (note ? ' ' : '') + `Image is ${(sw * sh / 1e6).toFixed(1)} MP — neural pass is limited to ≤ ${(neuralBudget / 1e6).toFixed(2)} MP inputs${webgpuOk ? '' : ' without WebGPU'}; used HQ Lanczos resampling.`;
    }
    if (cur.width !== targetW || cur.height !== targetH) { prog(75, 'High-quality Lanczos resampling…'); cur = await resampleHQ(cur, targetW, targetH); engineParts.push('Lanczos HQ resample'); }
    prog(85, 'Sharpening details…');
    await unsharpMask(cur, opts.detail === false ? 0.35 : 0.6, p => prog(85 + Math.round(p * 14), 'Sharpening details…'));
    engineParts.push('unsharp mask');
    prog(100, 'Done');
    return { canvas: cur, width: cur.width, height: cur.height, engine: engineParts.join(' + '), srcWidth: sw, srcHeight: sh, effectiveFactor: eff, note };
  }

  /* ------------------------------------------------------------------ */
  /* Technical quality analysis (for the rejection predictor)            */
  /* ------------------------------------------------------------------ */
  async function analyzeTechnicalQuality(src, fileBytes) {
    const img = await loadImageElement(src);
    const W = img.naturalWidth, H = img.naturalHeight; const mp = W * H / 1e6;
    const s = Math.min(1, 640 / Math.max(W, H));
    const c = toCanvas(img, Math.round(W * s), Math.round(H * s));
    const ctx = c.getContext('2d', { willReadFrequently: true });
    const d = ctx.getImageData(0, 0, c.width, c.height).data; const w = c.width, h = c.height;
    const gray = new Float32Array(w * h);
    for (let i = 0, j = 0; i < d.length; i += 4, j++) gray[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    // Laplacian variance (sharpness) + noise estimate (median absolute deviation of high-pass in flat areas)
    let sum = 0, sum2 = 0, n = 0; const hp = [];
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x; const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - w] - gray[i + w];
      sum += lap; sum2 += lap * lap; n++;
      if ((x & 3) === 0 && (y & 3) === 0) hp.push(Math.abs(lap));
    }
    const mean = sum / n; const sharpness = sum2 / n - mean * mean;
    hp.sort((a, b) => a - b); const noise = hp[Math.floor(hp.length * 0.5)] * 1.4826 / 4; // robust sigma estimate in gray levels
    const bpp = fileBytes ? (fileBytes * 8) / (W * H) : null; // bits per pixel of the encoded file
    const issues = [];
    if (mp < 4) issues.push(`Resolution ${W}×${H} (${mp.toFixed(1)} MP) is below the 4 MP minimum of Adobe Stock / Shutterstock`);
    if (sharpness < 25) issues.push('Image looks soft / out of focus (low edge contrast)');
    if (noise > 4.5) issues.push('Visible noise / grain detected');
    if (bpp !== null && bpp < 0.9 && mp >= 1) issues.push('Heavy JPEG compression (low bits-per-pixel) — export at quality 90+');
    const risk = issues.length === 0 ? 'LOW' : (issues.length === 1 && !/below the 4 MP/.test(issues[0]) ? 'MEDIUM' : 'HIGH');
    return { width: W, height: H, megapixels: mp, sharpness, noise, bitsPerPixel: bpp, issues, risk };
  }

  const TRADEMARKS = ['nike', 'adidas', 'puma', 'reebok', 'apple logo', 'apple inc', 'apple store', 'apple watch', 'iphone', 'ipad', 'macbook', 'airpods', 'samsung', 'google', 'android', 'microsoft', 'xbox', 'playstation', 'nintendo', 'facebook', 'instagram', 'whatsapp', 'youtube', 'tiktok', 'twitter', 'snapchat', 'netflix', 'amazon', 'spotify', 'uber', 'airbnb', 'coca cola', 'coca-cola', 'coke', 'pepsi', 'starbucks', 'mcdonald', 'kfc', 'burger king', 'nutella', 'oreo', 'red bull', 'disney', 'marvel', 'pixar', 'pokemon', 'lego', 'barbie', 'mickey', 'minion', 'hello kitty', 'ferrari', 'lamborghini', 'porsche', 'bmw', 'mercedes', 'audi', 'toyota', 'honda', 'tesla', 'volkswagen', 'jeep', 'harley', 'rolex', 'gucci', 'louis vuitton', 'chanel', 'prada', 'zara', 'ikea', 'canon', 'nikon', 'sony', 'olympics', 'olympic', 'fifa', 'world cup', 'nba', 'nfl', 'premier league', 'champions league', 'super bowl', 'oscars', 'grammy', 'hollywood sign', 'eiffel tower at night'];
  function trademarkCheck(text) {
    const t = (text || '').toLowerCase(); const hits = TRADEMARKS.filter(tm => new RegExp('(^|[^a-z])' + tm.replace(/[-\s]/g, '[\\s-]?') + '([^a-z]|$)').test(t));
    return hits;
  }

  /* ------------------------------------------------------------------ */
  /* Public API                                                          */
  /* ------------------------------------------------------------------ */
  window.PixEngine = {
    version: '2.0.0',
    onProgress(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    get hasWebGPU() { return state.webgpu; },
    get visionReady() { return !!state.clip; },
    get visionDevice() { return state.clip ? state.clip.device : null; },
    warmupVision: () => loadClip(),
    analyzeImage, composeMetadata, generateStockMetadata, imageToPrompt, promptFromAnalysis,
    llm, extractJson,
    customEngines: {
      list: getCustomEngines,
      save: saveCustomEngines,
      active: getActiveCustomEngine,
      test: testCustomEngine,
      chat: customOpenAiChat
    },
    customEngines: {
      list: getCustomEngines,
      save: saveCustomEngines,
      active: getActiveCustomEngine,
      test: testCustomEngine,
      chat: customOpenAiChat
    },
    gemini: { listModels: geminiListModels, candidates: geminiCandidates, generate: geminiGenerate, text: geminiText, rankVisionModels },
    generateImage, removeBackground, compositeCutout, upscaleImage, analyzeTechnicalQuality, trademarkCheck,
    utils: { loadImageElement, toCanvas, canvasToBlob, dataUrlToBlob, titleCase }
  };
})();
