Tailwind setup and notes

This project now uses Tailwind CSS v3. If you just checked out the repo, run:

1) Install node deps (will include tailwind/postcss):

   npm install

2) Start dev server (creates Tailwind runtime CSS):

   npm start

Notes
- The editor/linters may flag `@tailwind` and `@apply` rules in `src/index.css` as unknown because PostCSS processing is required. This is expected. The browser build will process Tailwind directives when you run `npm start`.
- To proxy frontend requests to the backend during development, add to `frontend/package.json`:

  "proxy": "http://localhost:8000"

and restart the dev server.
