import { useEffect, useRef, useState } from "react";

/* 32×32 SLVIS / Tensi sprite. 원본: design_handoff_slvis_redesign/prototypes/slvis-v4-bbs.jsx */
const TENSI_SPRITE: string[] = [
  "................................",
  "................................",
  "............HHHHHHHH............",
  "..........HHHhhhhhhHHH..........",
  "........HHhhhhhhhhhhhhHH........",
  ".......HhhhhhhhhhhhhhhhHH.......",
  "......HhhhhhffffffffhhhhH.......",
  "......HhhhhfffffffffffhhhH......",
  ".....HhhhhfffffffffffffhhhH.....",
  "....Hhhhhhfffffffffffffhhhhh....",
  "....Hhhhhffffffffffffffffhhh....",
  "....Hhhhfffeeffffffeefffffhh....",
  "....Hhhhfffeewfffffeewffffh.....",
  "....Hhhhhfffeeffffffeefffff.....",
  ".....Hhhhhfffffffffffffffh......",
  "......Hhhhffffppppppffffhh......",
  "......HhhhffffmmmmmmffffH.......",
  ".......Hhhhffffmmmmffffh........",
  "........Hhhhhffffffhhhh.........",
  ".........HhhhhhhhhHHH...........",
  "...........bbbbbbb..............",
  "..........bwwwwwwwb.............",
  ".........bwwwwwwwwwb............",
  "........bwwwwwwwwwwwb...........",
  "........bwwwwwwwwwwwb...........",
  "........bwwwwwwwwwwwb...........",
  ".........bwwwwwwwwwb............",
  "..........bbbbbbbbb.............",
  "................................",
  "................................",
  "................................",
  "................................",
];

const PAL: Record<string, string | null> = {
  ".": null,
  H: "#3A1E5F",
  h: "#5A2F85",
  f: "#FFE3C6",
  e: "#2C0E4A",
  p: "#FF92B8",
  m: "#B24061",
  b: "#FFD84D",
  w: "#FFFFFF",
};

type Props = {
  size?: number;
  animate?: boolean;
};

export default function Tensi({ size = 360, animate = true }: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [reducedMotion, setReducedMotion] = useState(false);

  // Live refs so the rAF loop can read the latest values without re-subscribing
  const gazeRef = useRef({ x: 0, y: 0, inside: false });
  const idleRef = useRef(false);
  const glitchRef = useRef(0); // 0=none, otherwise remaining ms
  const motionRef = useRef(animate);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const onChange = () => setReducedMotion(mq.matches);
    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, []);

  const doAnimate = animate && !reducedMotion;

  useEffect(() => {
    motionRef.current = doAnimate;
  }, [doAnimate]);

  // Pointer tracking (only when animating)
  useEffect(() => {
    if (!doAnimate) {
      gazeRef.current = { x: 0, y: 0, inside: false };
      return;
    }
    const onMove = (e: PointerEvent) => {
      gazeRef.current.x = e.clientX;
      gazeRef.current.y = e.clientY;
      gazeRef.current.inside = true;
    };
    const onLeave = () => {
      gazeRef.current.inside = false;
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerleave", onLeave);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerleave", onLeave);
    };
  }, [doAnimate]);

  // Idle detection (3.5s)
  useEffect(() => {
    if (!doAnimate) {
      idleRef.current = true;
      return;
    }
    let timer: ReturnType<typeof setTimeout>;
    const reset = () => {
      idleRef.current = false;
      clearTimeout(timer);
      timer = setTimeout(() => {
        idleRef.current = true;
      }, 3500);
    };
    reset();
    const evs: (keyof WindowEventMap)[] = [
      "pointermove",
      "keydown",
      "wheel",
      "touchstart",
    ];
    evs.forEach((ev) => window.addEventListener(ev, reset, { passive: true }));
    return () => {
      clearTimeout(timer);
      evs.forEach((ev) => window.removeEventListener(ev, reset));
    };
  }, [doAnimate]);

  // Glitch scheduler — 30-60s window, 200ms shift
  useEffect(() => {
    if (!doAnimate) return;
    let cancelled = false;
    const schedule = () => {
      const delay = 30000 + Math.random() * 30000;
      const t = setTimeout(() => {
        if (cancelled) return;
        glitchRef.current = 200;
        const end = setTimeout(() => {
          if (cancelled) return;
          glitchRef.current = 0;
          schedule();
        }, 200);
        (schedule as any)._end = end;
      }, delay);
      (schedule as any)._t = t;
    };
    schedule();
    return () => {
      cancelled = true;
      if ((schedule as any)._t) clearTimeout((schedule as any)._t);
      if ((schedule as any)._end) clearTimeout((schedule as any)._end);
    };
  }, [doAnimate]);

  // Canvas render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const host = wrapRef.current;
    if (!canvas || !host) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.imageSmoothingEnabled = false;

    const GRID = 32;
    const PX = size / GRID;
    let raf = 0;

    const draw = (t: number) => {
      ctx.clearRect(0, 0, size, size);

      // Screen backdrop
      ctx.fillStyle = "#061408";
      ctx.fillRect(0, 0, size, size);

      // Intra-sprite scanlines for medium+ sizes
      if (size > 150) {
        ctx.fillStyle = "rgba(0,0,0,0.12)";
        for (let y = 0; y < size; y += 3) {
          ctx.fillRect(0, y, size, 1);
        }
      }

      const g = gazeRef.current;
      const idl = idleRef.current;
      const animating = motionRef.current;

      // Normalized gaze vector (-1..1) relative to canvas center
      let lx = 0;
      let ly = 0;
      if (animating && g.inside) {
        const r = host.getBoundingClientRect();
        lx = Math.max(
          -1,
          Math.min(1, (g.x - (r.left + r.width / 2)) / (r.width / 2)),
        );
        ly = Math.max(
          -1,
          Math.min(1, (g.y - (r.top + r.height / 2)) / (r.height / 2)),
        );
      }

      const tilt = animating && !idl ? Math.round(lx * 1.4) : 0;
      const dyh = animating && !idl ? Math.round(ly * 0.6) : 0;
      const blink = animating && !idl && Math.sin(t / 600) > 0.96;

      // Glitch: shift rows 1-2 horizontally
      const glitchActive = animating && glitchRef.current > 0;
      if (glitchActive) glitchRef.current = Math.max(0, glitchRef.current - 16);
      const glitchShift = glitchActive ? (Math.random() > 0.5 ? 2 : -2) : 0;

      for (let y = 0; y < GRID; y++) {
        for (let x = 0; x < GRID; x++) {
          const k = TENSI_SPRITE[y][x];
          const col = PAL[k];
          if (!col) continue;
          const dx = y >= 2 && y <= 19 ? tilt : 0;
          const dy = y >= 2 && y <= 19 ? dyh : 0;
          const gx = glitchActive && (y === 10 || y === 11) ? glitchShift : 0;
          let c1 = col;
          if (blink && y >= 11 && y <= 13 && (k === "e" || k === "w")) {
            c1 = PAL.f as string;
          }
          ctx.fillStyle = c1;
          ctx.fillRect((x + dx + gx) * PX, (y + dy) * PX, PX + 0.5, PX + 0.5);
        }
      }

      // Vignette + sweep (larger sizes only)
      if (size > 150) {
        const vg = ctx.createRadialGradient(
          size / 2,
          size / 2,
          size * 0.4,
          size / 2,
          size / 2,
          size * 0.78,
        );
        vg.addColorStop(0, "rgba(0,0,0,0)");
        vg.addColorStop(1, "rgba(0,0,0,0.45)");
        ctx.fillStyle = vg;
        ctx.fillRect(0, 0, size, size);

        // 상시 스윕 스캔라인은 주변시 피로도 큼 → 제거. vignette만으로 충분.
      }

      if (animating) {
        raf = requestAnimationFrame(draw);
      }
    };

    // Kick off
    if (doAnimate) {
      raf = requestAnimationFrame(draw);
    } else {
      draw(0);
    }

    return () => cancelAnimationFrame(raf);
  }, [size, doAnimate]);

  return (
    <div
      ref={wrapRef}
      aria-label="SLVIS 픽셀 캐릭터"
      role="img"
      style={{
        position: "relative",
        width: size,
        height: size,
        background: "#061408",
        borderRadius: size > 150 ? 14 : 3,
        boxShadow:
          size > 150
            ? "inset 0 0 40px rgba(0,0,0,0.6), 0 0 14px rgba(155,229,173,0.08)"
            : "none",
        overflow: "hidden",
        flexShrink: 0,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: size,
          height: size,
          display: "block",
          imageRendering: "pixelated",
        }}
      />
    </div>
  );
}
