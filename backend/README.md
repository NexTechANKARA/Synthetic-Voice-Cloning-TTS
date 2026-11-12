# Voice Cloner backend

A small FastAPI app that runs the five trained models (from `vc_lib/` and `checkpoints/`) on a free
Hugging Face Space (CPU). Built in the next step.

| Endpoint | What it does |
| --- | --- |
| `GET /voices` | The built-in voices, with gender and accent |
| `POST /embed` | Audio clip in, 64-number voice embedding out. Nothing is stored |
| `POST /speak` | Text, voice and speed in, WAV audio out |

Every request needs a Supabase sign-in token. Limits: 30 generations per user per hour, 1000 characters
per request.
