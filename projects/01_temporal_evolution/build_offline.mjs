import { readFile, writeFile } from 'node:fs/promises'
import { build } from 'esbuild'

const corpus = JSON.parse(await readFile(new URL('./public/data/corpus.json', import.meta.url), 'utf8'))
const atlas = JSON.parse(await readFile(new URL('./public/data/templates.json', import.meta.url), 'utf8'))
const artworkUrl = new URL('./artwork.js', import.meta.url)

// Classic script globals work from file:// in every major desktop browser.
// The scientific data remains in separate, inspectable source JSON as well.
await writeFile(
  new URL('./offline-data.js', import.meta.url),
  `globalThis.__NEURO_CORPUS__=${JSON.stringify(corpus)};\nglobalThis.__NEURO_ATLAS__=${JSON.stringify(atlas)};\n`,
)

await build({
  entryPoints: [new URL('./src/main.ts', import.meta.url).pathname],
  outfile: artworkUrl.pathname,
  bundle: true,
  format: 'iife',
  platform: 'browser',
  target: ['es2020'],
  minify: true,
  legalComments: 'none',
  sourcemap: false,
  loader: { '.css': 'css' },
  logLevel: 'warning',
})

// Some upstream shader chunks contain trailing spaces inside template literals.
// Normalize the generated artifact so clean builds also pass Git whitespace checks.
const artwork = await readFile(artworkUrl, 'utf8')
await writeFile(artworkUrl, artwork.replace(/[ \t]+$/gm, ''))

console.log(`Offline artwork packaged: ${corpus.papers.length.toLocaleString()} papers / ${atlas.entries.length} forms`)
