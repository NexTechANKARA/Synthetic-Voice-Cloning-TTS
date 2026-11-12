# Voice Cloner website

Plain HTML, CSS and JavaScript with no build step, hosted on Netlify (`netlify.toml` at the repository
root). The five models run in the visitor's browser with ONNX Runtime Web, so there is no server: text and
voice recordings never leave the device. Sign-in uses Supabase.

| Page | What it does |
| --- | --- |
| `index.html` | The intro (its counter follows the real model download), then sign in, create an account or reset a password |
| `studio.html` | Type text, choose one of 80 voices or add your own by recording or upload, hear it and download a WAV |
| `results.html` | Training curves and evaluation numbers. Public, no sign-in |

| File | Contents |
| --- | --- |
| `js/engine.js` | The pipeline from `vc_lib/infer.py`, step for step, on ONNX Runtime Web |
| `js/landscape.js` | The spectrogram landscape and the intro drawing |
| `js/intro.js`, `js/signin.js`, `js/studio.js`, `js/results.js` | One script per page (plus the intro) |
| `js/auth.js`, `config.js` | Supabase sign-in and its public settings |
| `models/` | The ONNX models and `pipeline.json`, made by `python tools/export_onnx.py` |
| `data/results.json` | Training history for the results page |

To try it locally, serve this folder (for example `python -m http.server` inside `frontend/`) and open
http://localhost:8000. Sign-in works once the address is added to Supabase's allowed redirect URLs.
