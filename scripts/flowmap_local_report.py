"""View local Flow Map losses and MP4 previews without W&B or other dependencies.

python scripts/flowmap_local_report.py RUN_DIR --serve 8765
Or omit --serve to write a standalone dashboard.html in RUN_DIR.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # The trainer may still be writing the final record.
    return rows


def chart(title, points):
    points = [(x, a, b, c) for x, a, b, c in points
              if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, a, b, c))]
    if not points:
        return ''
    x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
    lo, hi = min(p[2] for p in points), max(p[3] for p in points)
    pad = max((hi-lo)*.08, abs(hi)*.01, 1e-12)
    lo -= pad; hi += pad
    xy = lambda x, y: f'{65+(x-x0)/max(x1-x0,1)*500:.2f},{195-(y-lo)/(hi-lo)*155:.2f}'
    mean = ' '.join(xy(x,a) for x,a,_,_ in points)
    band = ' '.join([xy(x,b) for x,_,b,_ in points] + [xy(x,c) for x,_,_,c in reversed(points)])
    dots = ''.join(f'<circle cx="{xy(x,a).split(",")[0]}" cy="{xy(x,a).split(",")[1]}" r="2" fill="#2563eb"/>'
                   for x,a,_,_ in points) if len(points)<20 else ''
    return f'''<section><h3>{escape(title)}</h3><svg viewBox="0 0 600 240" role="img" aria-label="{escape(title)}">
    <path d="M65 40V195H565" fill="none" stroke="#999"/>
    <polygon points="{band}" fill="#dbeafe"/><polyline points="{mean}" fill="none" stroke="#2563eb" stroke-width="2"/>{dots}
    <text x="3" y="46">{hi:.3g}</text><text x="3" y="195">{lo:.3g}</text>
    <text x="65" y="216">{x0}</text><text x="530" y="216">{x1}</text>
    <text x="235" y="236">Optimizer update</text></svg></section>'''


def render(root):
    # Keep the newest record if a recovery replays an optimizer update.
    by_step = {r['step']:r for r in read_rows(root/'train_update_metrics.jsonl') if 'step' in r}
    rows = [by_step[s] for s in sorted(by_step)]
    latest = rows[-1] if rows else {}
    manifest, status = read_json(root/'manifest.json'), read_json(root/'segment_status.json')
    horizon = status.get('max_steps', manifest.get('max_steps', '?'))
    metrics = sorted({k for r in rows for k in r.get('metrics', {})}, key=lambda k:(k!='loss',k))
    plots = []
    for metric in metrics:
        values = []
        for r in rows:
            v = r.get('metrics', {}).get(metric)
            if v:
                values.append((r['step'],v['mean'],v.get('microbatch_min',v['mean']),v.get('microbatch_max',v['mean'])))
        plots.append(chart(metric, values))
    for key, label in [('grad_norm','Gradient norm'),('learning_rate','Learning rate'),
                       ('peak_allocated_gib','Peak GPU allocation (GiB)'),('peak_reserved_gib','Peak GPU reservation (GiB)')]:
        plots.append(chart(label, [(r['step'],r[key],r[key],r[key]) for r in rows if key in r]))
    saved = sorted((root/'checkpoints/state').glob('step_*/trainer_state.json'))
    checkpoint = saved[-1].parent.name if saved else 'None yet'
    videos = sorted((root/'eval').glob('*.mp4'))
    preview_rows = read_rows(root/'preview_metrics.jsonl')
    latest_preview = max((r.get('step',0) for r in preview_rows), default=None)
    if latest_preview is not None:
        videos = [p for p in videos if f'step_{latest_preview:06d}_' in p.name]
    media = ''.join(f'<section><h3>{escape(p.name)}</h3><video controls preload="none" src="{quote(str(p.relative_to(root)))}"></video></section>'
                    for p in videos[-12:])
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Flow Map training</title><style>
    body{{font:16px system-ui;margin:2rem auto;max-width:1250px;padding:0 1rem;color:#172033;background:#f5f7fb}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}}
    section{{background:white;border:1px solid #dce2ec;border-radius:10px;padding:1rem;min-width:0}}
    h3{{font-size:16px;overflow-wrap:anywhere}}svg,video{{width:100%}}svg text{{font:12px system-ui}}
    code{{overflow-wrap:anywhere}}button{{padding:.5rem 1rem}}small{{color:#586477}}
    </style><h1>Flow Map training</h1><p><code>{escape(root.name)}</code></p>
    <p><b>Update {latest.get('step',0)} / {horizon}</b> · Latest full checkpoint: {escape(checkpoint)} · Batch: {latest.get('examples',manifest.get('effective_batch','?'))}</p>
    <p><button onclick="location.reload()">Refresh</button> <small>Rendered {stamp}. Refresh to read new results.</small></p>
    <p>Loss curves show update means; shading spans the microbatch minimum and maximum.</p>
    <div class="grid">{''.join(plots) if rows else '<section>No training metrics recorded yet. Check Slurm for queue status.</section>'}</div>
    <h2>Latest future previews</h2><p>Raw weights at 1, 2 and 4 NFEs. Training-set examples, not benchmark success rates.</p>
    <div class="grid">{media or '<section>No preview videos yet. The first previews are saved after the two-update hardware check.</section>'}</div>
    </html>'''


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        root = Path(self.directory).resolve()
        path = unquote(urlsplit(self.path).path)
        if path in ('/', '/dashboard.html'):
            body = render(root).encode()
            self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif path.startswith('/eval/'):
            file = (root/path.lstrip('/')).resolve()
            if file.is_relative_to(root/'eval') and file.suffix=='.mp4' and file.is_file():
                super().do_GET()
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_HEAD(self):
        self.send_error(405)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--serve',type=int,metavar='PORT')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if not root.is_dir():parser.error('Run directory does not exist')
    if args.serve:
        server = ThreadingHTTPServer(('127.0.0.1',args.serve),partial(Handler,directory=str(root)))
        print(f'Local dashboard: http://127.0.0.1:{args.serve} — Ctrl+C to stop',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()
    else:
        output = root/'dashboard.html';output.write_text(render(root))
        print(output)


if __name__=='__main__':main()
