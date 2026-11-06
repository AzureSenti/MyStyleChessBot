#!/usr/bin/env python3
"""A tiny local web UI to play against your fine-tuned Maia3 model.

Runs a localhost HTTP server + your trained engine (maia3-uci UCI wrapper).
Open http://localhost:8765 in a browser after starting.

Usage:
  .venv/bin/python code/playui.py
  .venv/bin/python code/playui.py --checkpoint finetunes/maia3-79m/best.pt --port 8765
  .venv/bin/python code/playui.py --model maia3-79m    # play the base (pretrained) model
"""

from __future__ import annotations

import argparse
import chess
import json
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------
# UCI engine wrapper (blocking calls, protected by the engine's own lock)
# --------------------------------------------------------------------------------------
class Engine:
    def __init__(self, engine_args: list[str]):
        self.lock = threading.Lock()
        self.p = subprocess.Popen(engine_args, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)
        self._send("uci")
        self._read_ok("uciok")

    def _send(self, cmd: str):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _read_ok(self, token: str, timeout: float = 600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.strip() == token:
                return

    def set_option(self, name, value):
        with self.lock:
            self._send(f"setoption name {name} value {value}")

    def new_game(self):
        with self.lock:
            self._send("ucinewgame")
            self._send("isready")
            self._read_ok("readyok")

    def go(self, moves: list[str]) -> dict:
        """Feed position startpos moves ...; returns {'bestmove', 'top5', 'ms'}."""
        pos = "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        with self.lock:
            self._send(pos)
            self._send("go")
            top5: list[dict] = []
            best = None
            t0 = time.time()
            while True:
                line = self.p.stdout.readline()
                if not line:
                    raise RuntimeError("engine died")
                line = line.strip()
                if line.startswith("info"):
                    parts = line.split()
                    try:
                        rank = int(parts[parts.index("multipv") + 1])
                        cp = int(parts[parts.index("score") + 2])
                        w = parts[parts.index("wdl") + 1:]
                        wdl = (int(w[0]), int(w[1]), int(w[2]))
                        pv = parts[parts.index("pv") + 1]
                        if len(top5) < 5 and all(e["rank"] != rank for e in top5):
                            top5.append({"rank": rank, "cp": cp, "wdl": wdl, "move": pv})
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("bestmove"):
                    tok = line.split()
                    best = tok[1] if len(tok) > 1 and tok[1] != "(none)" else None
                    break
                if time.time() - t0 > 300:
                    break
            return {"bestmove": best, "top5": top5, "ms": int((time.time() - t0) * 1000)}


# --------------------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------------------
class GameServer:
    def __init__(self, engine_args, user_elo=1460, temperature=0.4):
        self.engine = Engine(engine_args)
        self.lock = threading.Lock()
        self.reset("white", user_elo, temperature)

    def reset(self, color: str, elo: int, temperature: float):
        self.board = chess.Board()
        self.moves: list[str] = []
        self.my_color = chess.WHITE if color == "white" else chess.BLACK
        self.top5: list[dict] | None = None
        self.last_ms: int | None = None
        self.error: str | None = None
        self.engine.set_option("Temperature", float(temperature))
        self.engine.set_option("TopP", 1.0)
        self.engine.set_option("SelfElo", int(elo))
        self.engine.set_option("OppoElo", int(elo))
        self.engine.new_game()

    def to_move_user(self):
        return (self.board.turn == self.my_color) and not self.board.is_game_over(claim_draw=True)

    def state(self) -> dict:
        return {
            "fen": self.board.fen(),
            "moves": self.moves,
            "your_turn": self.to_move_user(),
            "your_color": "white" if self.my_color == chess.WHITE else "black",
            "top5": self.top5,
            "engine_ms": self.last_ms,
            "error": self.error,
            "over": self.board.is_game_over(claim_draw=True),
            "result": self.board.result(claim_draw=True) if
                      self.board.is_game_over(claim_draw=True) else None,
        }

    def apply_engine_move(self, mv_uci: str | None):
        if not mv_uci:
            return
        try:
            mv = chess.Move.from_uci(mv_uci)
        except ValueError:
            return
        if mv in self.board.legal_moves:
            self.board.push(mv)
            self.moves.append(mv.uci())

    def engine_turn(self):
        r = self.engine.go(self.moves)
        self.top5 = r["top5"][:5]
        self.last_ms = r["ms"]
        self.apply_engine_move(r["bestmove"])

    # -- API ops --
    def play(self, from_sq, to_sq) -> dict:
        with self.lock:
            self.error = None
            if not self.to_move_user():
                return {"ok": False, "error": "not your turn"}
            try:
                mv = chess.Move.from_uci(from_sq + to_sq)
                if mv not in self.board.legal_moves:
                    mv = chess.Move.from_uci(from_sq + to_sq + "q")  # promotion: queen mặc định
            except ValueError:
                return {"ok": False, "error": "invalid move", "state": self.state()}
            if mv not in self.board.legal_moves:
                self.error = "invalid move"
                return {"ok": False, "error": "invalid move", "state": self.state()}
            self.board.push(mv)
            self.moves.append(mv.uci())
            self.top5 = None
            if not self.board.is_game_over(claim_draw=True):
                self.engine_turn()
            return {"ok": True, "state": self.state()}

    def new_game(self, color: str, elo: int, temperature: float) -> dict:
        with self.lock:
            self.reset(color.lower(), elo, temperature)
            if self.my_color == chess.BLACK:
                self.engine_turn()
            return {"ok": True, "state": self.state()}

    def undo(self) -> dict:
        with self.lock:
            self.error = None
            if not self.moves:
                return {"ok": False, "state": self.state()}
            if self.to_move_user() and len(self.moves) >= 2:
                del self.moves[-2:]   # undo your last move + engine reply
            else:
                del self.moves[-1:]
            self.board = chess.Board()
            for mv in self.moves:
                self.board.push(chess.Move.from_uci(mv))
            self.top5 = None
            return {"ok": True, "state": self.state()}


# --------------------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    game: GameServer

    def do_GET(self):
        if self.path == "/state":
            self._json(self.game.state())
        elif self.path == "/":
            self._html(PAGE)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        data = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/move":
            out = self.game.play(data["from"], data["to"])
        elif self.path == "/new":
            out = self.game.new_game(data.get("color", "white"),
                                     int(data.get("elo", 1460)),
                                     float(data.get("temperature", 0.4)))
        elif self.path == "/undo":
            out = self.game.undo()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self._json(out)

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, page):
        body = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a):
        pass


# --------------------------------------------------------------------------------------
# UI (unicode board, no external CDN needed)
# --------------------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Maia3 · bot của bạn</title>
<style>
 :root{--light:#ebecd0;--dark:#739552;--bg:#262421;--fg:#e8e6e3;--hl:#cdd26a;--panel:#3d3a35}
 body{background:var(--bg);color:var(--fg);font-family:system-ui,'Segoe UI',sans-serif;margin:0;padding:16px}
 h1{font-size:20px;margin:0 0 6px}
 sub{color:#9a9691;font-size:12.5px;margin-bottom:14px;display:block}
 .wrap{display:flex;flex-wrap:wrap;gap:22px;align-items:flex-start}
 .boardcol{width:min(88vmin,560px)}
 .grid{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1;border:4px solid #4545;user-select:none;cursor:pointer}
 .sq{display:flex;align-items:center;justify-content:center;aspect-ratio:1;font-size:min(9vmin,6.6vmin);line-height:1}
 .sq.l{background:var(--light)} .sq.d{background:var(--dark)}
 .sq.sel{box-shadow:inset 0 0 0 4px var(--hl)}
 .piece.w{color:#fcfbfa;text-shadow:0 1px 2px #00000088,0 0 2px #0006}
 .piece.b{color:#151311;text-shadow:0 0 2px #ffffff44}
 .panel{flex:1 1 250px;max-width:320px}
 .card{background:var(--panel);border-radius:10px;padding:12px 14px;margin-bottom:12px}
 button{background:#4a7dcc;border:0;color:#fff;border-radius:6px;padding:7px 13px;margin:2px 2px 2px 0;cursor:pointer;font-size:14px}
 button.ghost{background:#4a4a45}
 input{background:#2a2825;color:var(--fg);border:1px solid #55504a;border-radius:5px;padding:5px 8px;font-size:13px}
 .row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:6px 0}
 .mt{font-size:12.5px;color:#aca9a3}
 .t5{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.7}
 .badge{display:inline-block;background:#00000040;padding:1px 7px;border-radius:20px;margin-right:6px}
 .mvlist{font-family:ui-monospace,Menlo,monospace;font-size:12px;max-height:130px;overflow:auto;color:#c9c6c1;white-space:pre}
 #err{color:#ff9a9a;font-size:13px;min-height:18px}
 #th{color:#e2c871;font-size:13px}
 #status{color:#aca9a3;font-size:13.5px}
</style></head><body>
<h1>Maia3 · <span style="color:#e2c871">bot fine-tune theo phong cách của bạn</span></h1>
<sub>Bạn đang cầm <b id="yc">?</b> · dữ liệu học: 1 879 ván rapid của bạn · <span id="eng"></span></sub>
<div class="wrap">
 <div class="boardcol">
   <div class="grid" id="grid"></div>
   <div class="row">
     <button id="bNewW">Ván mới (Trắng)</button>
     <button id="bNewB">Ván mới (Đen)</button>
     <button class="ghost" id="bUndo">Bỏ 1 nước</button>
   </div>
   <div class="row">
     <label class="mt">SelfElo <input id="elo" type="number" value="1460" style="width:70px"></label>
     <label class="mt">Temperature <input id="temp" type="number" step="0.1" min="0" max="2" value="0.4" style="width:60px"></label>
   </div>
   <div id="err"></div><div id="th">🤔 model đang tính nước…</div>
 </div>
 <div class="panel">
   <div class="card"><b>5 nước model ưu tiên nhất</b><div class="t5" id="t5" style="margin-top:8px">—</div>
     <div class="mt" style="margin-top:6px">MultiPV — xem model "nghĩ về nước của bạn" thế nào trong thế hiện tại.</div></div>
   <div class="card"><b>Nước đi</b><div class="mvlist" id="mv"></div></div>
   <div class="card"><div id="status">Đang kết nối…</div></div>
 </div>
</div>
<script>
const P={n:'\u265E',b:'\u265D',r:'\u265C',q:'\u265B',k:'\u265A',p:'\u265F'};
const FILES='abcdefgh';
let S=null, sel=null, flipped=false, pending=false;
const $=id=>document.getElementById(id);

function render(){
  const g=$('grid'); g.innerHTML='';
  const rows=S.fen.split(' ')[0].split('/');
  for(let vr=0; vr<8; vr++){
    const rank = flipped ? vr+1 : 8-vr;            // rank 1..8
    const row = rows[8-rank];
    for(let vc=0; vc<8; vc++){
      const fi = flipped ? 7-vc : vc;               // file 0..7 (a..h)
      const ch = row[fi] || ' ';
      const div=document.createElement('div');
      div.className='sq '+(((fi+rank)%2===1)?'d':'l');
      div.dataset.sq=FILES[fi]+rank;
      if(!/[0-9]/.test(ch)) div.innerHTML=`<span class="piece ${/^[A-Z]/.test(ch)?'w':'b'}">${P[ch.toLowerCase()]}</span>`;
      if(sel===div.dataset.sq) div.classList.add('sel');
      div.onclick=()=>click(div.dataset.sq);
      g.appendChild(div);
    }
  }
}

function click(sq){
  if(!S || !S.your_turn || pending) return;
  if(!sel){sel=sq;render();return;}
  if(sel===sq){sel=null;render();return;}
  pending=true; $('th').textContent='⏳ gửi nước…';
  post('/move',{from:sel,to:sq}).then(()=>{sel=null;pending=false;refresh();});
}

function mvText(mvs){
  let out=[];
  for(let i=0;i<mvs.length;i+=2) out.push(`${Math.floor(i/2)+1}. ${mvs[i]} ${mvs[i+1]||''}`);
  return out.join('\n');
}

async function post(u,d){
  const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  return r.json();
}

async function refresh(){
  try{
    S = await(await fetch('/state')).json();
    $('yc').textContent = S.your_color==='white' ? 'Trắng' : 'Đen';
    $('mv').textContent = mvText(S.moves);
    $('t5').innerHTML = S.top5 ? S.top5.map(t=>{
        const [w,d,l]=t.wdl;
        return `<span class="badge">#${t.rank}</span><b>${t.move}</b> · WDL ${w/10}/${d/10}/${l/10}`;
      }).join('<br>') : (S.moves.length? '—' : '—');
    $('th').textContent = (!S.over && !S.your_turn) ? '🤔 model đang tính nước…' : '';
    $('status').textContent = S.over
      ? `Ván kết thúc: ${S.result||''} — bấm “Ván mới” để chơi lại.`
      : (pending ? 'Đang xử lý…' : (S.your_turn ? '♟ Tới lượt bạn — bấm quân rồi bấm đích.' : ''));
    $('err').textContent = S.error || '';
    render();
  }catch(e){ $('status').textContent = 'Server chưa sẵn sàng…'; }
}

$('bNewW').onclick=()=>{sel=null;post('/new',{color:'white',elo:+$('elo').value,temperature:+$('temp').value}).then(refresh);};
$('bNewB').onclick=()=>{sel=null;post('/new',{color:'black',elo:+$('elo').value,temperature:+$('temp').value}).then(refresh);};
$('bUndo').onclick=()=>{sel=null;post('/undo',{}).then(refresh);};
refresh();
setInterval(()=>{ if(!pending) refresh(); }, 1500);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="finetunes/maia3-79m/best.pt")
    ap.add_argument("--model", default="maia3-79m", help="Alias for architecture flags")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--elo", type=int, default=1460, help="Starting SelfElo/OppoElo")
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    engine_bin = REPO_ROOT / ".venv" / "bin" / "maia3-uci"
    if not engine_bin.exists():
        engine_bin = shutil.which("maia3-uci") or "maia3-uci"
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    engine_args = [str(engine_bin), "--model", args.model,
                   "--checkpoint-path", str(ckpt),
                   "--use-uci-history", "--temperature", str(args.temperature)]
    print("engine:", " ".join(engine_args), flush=True)
    Handler.game = GameServer(engine_args, user_elo=args.elo, temperature=args.temperature)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n✅ Mở http://localhost:{args.port} để đấu với bot của bạn!", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
