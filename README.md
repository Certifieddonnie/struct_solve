# struct_solve

This repository contains a **static front‑end** for the Structural Analysis Pro application.  The client lives in the `frontend/` directory and uses a lightweight build system (Vite) so it can be hosted on services like Render.

## Frontend build (Vite)

The front‑end has been converted to a Vite project.  Environment variables such as the API endpoint are injected at build time and should never be hard‑coded in source files.

### Getting started locally

```bash
cd frontend
npm install          # first time only
npm run dev          # start development server (http://localhost:5173)
```

Configuration is done with a `.env` file in the `frontend` folder.  Copy the example and set your values:

```bash
cp frontend/.env.example frontend/.env
# edit .env and set VITE_API_URL to your back‑end URL
```

The client reads the API URL like this in `app.js`:

```js
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';
```

### Building for production

```bash
cd frontend
npm run build        # output will be placed in frontend/dist
```

The `dist` directory contains static HTML/CSS/JS that can be served by any web server.

## Deploying to Render

1. **Create a new Static Site** on Render and connect your GitHub repo.
2. Set the **build command** to:
   ```bash
   cd frontend && npm install && npm run build
   ```
3. Set the **publish directory** to `frontend/dist`.
4. Add any required environment variables under Render's **Environment** tab, e.g. `VITE_API_URL`.
5. Make sure `.env` is listed in `.gitignore` (it already is) so secrets are not committed.

Render will run the build, produce the `dist` output and serve the files.  Your API URL is hidden because it's supplied at build time via Render's environment settings.

---
