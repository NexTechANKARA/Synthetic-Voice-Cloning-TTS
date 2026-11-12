# Voice Cloner frontend

Plain HTML, CSS and JavaScript with no build step, hosted on Netlify. Built in the next step.

| Page | What it shows |
| --- | --- |
| Sign in | The intro, then email or Google sign-in (Supabase) |
| Studio | Type text, pick or add a voice, hear and download the result |
| Model results | Training curves and evaluation numbers. Public, no sign-in |

`config.js` holds the public settings: the Supabase project URL and its publishable key (safe to share;
it only allows what Supabase's sign-in rules allow), and the backend URL once the Space is live.
