# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

EXPECTED_EPOCHS = 40
EXPS = ["E0", "E1", "E2", "E5", "E7", "E5_oof", "E6"]
EXP_CKPT = {
    "E0": "runs/rgb_baseline/E0_rgb",
    "E1": "runs/bcp_vis/E1_zero",
    "E2": "runs/bcp_vis/E2_random",
    "E5": "runs/bcp_vis/E5_bcp",
    "E7": "runs/bcp_vis/E7_oracle",
    "E5_oof": "runs/bcp_vis/E5_oof",
    "E6": "runs/bcp_vis/E6_altitude",
}
CHAINS = ["run_chain_phase2.sh", "run_chain3_v4.sh", "resume_chain.sh"]

ERR_PAT = re.compile(r"Traceback|CUDA out of memory|RuntimeError|Killed|\[FAIL\]",
                     re.IGNORECASE)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def chains_alive():
    ps = sh("ps -eo args")
    lines = ps.splitlines()
    return [c for c in CHAINS if any(("bash scripts/" + c) in l for l in lines)]


def exp_progress(root: Path):
    rows = []
    for e in EXPS:
        d = root / EXP_CKPT[e]
        m = d / "metrics.csv"
        n = 0
        best = ""
        if m.exists():
            lines = [l for l in m.read_text(encoding="utf-8").splitlines() if l.strip()]
            n = max(0, len(lines) - 1)
            if n:
                try:
                    import csv as _csv
                    r = list(_csv.DictReader(m.open(encoding="utf-8")))
                    bi = max(range(len(r)), key=lambda i: float(r[i]["dice"]))
                    best = "IoU=%.4f Dice=%.4f" % (float(r[bi]["iou"]),
                                                   float(r[bi]["dice"]))
                except Exception:
                    pass
        done = (d / "best.pth").exists() and n >= EXPECTED_EPOCHS
        rows.append((e, n, done, best))
    return rows


def pending(rows):
    return [r[0] for r in rows if not r[2]]


def last_step(root: Path):
    out = []
    for lg in ("logs/phase2.log", "logs/phase3.log", "logs/resume.log"):
        p = root / lg
        if p.exists():
            lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            if lines:
                out.append("%s: %s" % (Path(lg).name, lines[-1]))
    return out


def scan_errors(root: Path, seen: dict):
    new = []
    for p in sorted((root / "logs").glob("*.log")):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hits = [l for l in txt.splitlines() if ERR_PAT.search(l)]
        key = p.name
        if len(hits) > seen.get(key, 0):
            for l in hits[seen.get(key, 0):]:
                new.append("%s | %s" % (key, l.strip()[:140]))
            seen[key] = len(hits)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--auto-resume", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    seen = {}

    print("watchdog started interval=%ss auto_resume=%s" % (a.interval, a.auto_resume),
          flush=True)
    while True:
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            alive = chains_alive()
            rows = exp_progress(root)
            pend = pending(rows)
            disk = sh("df -h /root/autodl-tmp | tail -1")
            gpu = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,"
                     "power.draw --format=csv,noheader")
            nprior = sh("ls %s/datasets/EVD4UAV_processed/oof_priors/*.png 2>/dev/null "
                        "| wc -l" % root)
            errs = scan_errors(root, seen)

            lines = []
            lines.append("# BCP-Vis 运行状态（由 tools/watchdog.py 自动更新）")
            lines.append("")
            lines.append("最后更新：%s" % ts)
            lines.append("")
            lines.append("## 执行链")
            lines.append("")
            if alive:
                for c in alive:
                    lines.append("- 运行中：`%s`" % c)
            else:
                lines.append("- **无执行链在运行**")
            for s in last_step(root):
                lines.append("- %s" % s)
            lines.append("")
            lines.append("## 实验进度（目标 %d epoch）" % EXPECTED_EPOCHS)
            lines.append("")
            lines.append("| 实验 | epoch | 状态 | 最佳 |")
            lines.append("|---|---|---|---|")
            for e, n, done, best in rows:
                lines.append("| %s | %d | %s | %s |" %
                             (e, n, "完成" if done else "进行中/未开始", best or "-"))
            lines.append("")
            lines.append("## 资源")
            lines.append("")
            lines.append("```")
            lines.append("磁盘: %s" % disk)
            lines.append("GPU : %s" % gpu)
            lines.append("先验文件: %s" % nprior)
            lines.append("```")
            lines.append("")
            if errs:
                lines.append("## 新出现的错误")
                lines.append("")
                for e in errs[-20:]:
                    lines.append("- `%s`" % e)
                lines.append("")
            (root / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

            if not alive and pend and a.auto_resume:
                print("[%s] 链已停止，未完成实验=%s -> 拉起 resume_chain.sh"
                      % (ts, pend), flush=True)
                subprocess.Popen(
                    ["setsid", "nohup", "bash", "scripts/resume_chain.sh"],
                    cwd=str(root),
                    stdout=open(root / "logs" / "resume.out", "a"),
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True)
            if errs:
                for e in errs[-5:]:
                    print("[%s] ERR %s" % (ts, e), flush=True)
            print("[%s] alive=%s pending=%s" % (ts, alive, pend), flush=True)
        except Exception as ex:
            print("watchdog error:", type(ex).__name__, ex, flush=True)
        time.sleep(a.interval)

if __name__ == "__main__":
    main()
