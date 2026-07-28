"use client";

import { useEffect, useRef } from "react";

type Point = {
  x: number;
  y: number;
  ox: number;
  oy: number;
  phase: number;
};

const COLS = 14;
const ROWS = 9;

export function MeshCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let points: Point[] = [];
    let raf = 0;
    let visible = document.visibilityState === "visible";
    let width = 0;
    let height = 0;
    let dpr = 1;

    const buildGrid = () => {
      const next: Point[] = [];
      for (let row = 0; row <= ROWS; row += 1) {
        for (let col = 0; col <= COLS; col += 1) {
          const ox = (col / COLS) * width;
          const oy = (row / ROWS) * height;
          next.push({
            x: ox,
            y: oy,
            ox,
            oy,
            phase: (col * 0.55 + row * 0.85) % (Math.PI * 2),
          });
        }
      }
      points = next;
    };

    const resize = () => {
      const parent = canvas.parentElement;
      width = parent?.clientWidth ?? window.innerWidth;
      height = parent?.clientHeight ?? window.innerHeight;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      buildGrid();
      draw(0);
    };

    const displace = (t: number) => {
      const amp = Math.min(width, height) * 0.012;
      for (const p of points) {
        p.x = p.ox + Math.sin(t * 0.00055 + p.phase) * amp;
        p.y = p.oy + Math.cos(t * 0.0004 + p.phase * 1.3) * amp;
      }
    };

    const draw = (t: number) => {
      if (!reduceMotion) {
        displace(t);
      }

      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(154, 156, 163, 0.2)";
      ctx.lineWidth = 1;

      const cols = COLS + 1;

      ctx.beginPath();
      for (let row = 0; row < ROWS; row += 1) {
        for (let col = 0; col < COLS; col += 1) {
          const i = row * cols + col;
          const a = points[i];
          const b = points[i + 1];
          const c = points[i + cols];
          const d = points[i + cols + 1];

          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.lineTo(c.x, c.y);
          ctx.closePath();

          ctx.moveTo(b.x, b.y);
          ctx.lineTo(d.x, d.y);
          ctx.lineTo(c.x, c.y);
          ctx.closePath();
        }
      }
      ctx.stroke();
    };

    const tick = (t: number) => {
      if (visible && !reduceMotion) {
        draw(t);
        raf = requestAnimationFrame(tick);
      }
    };

    const onVisibility = () => {
      visible = document.visibilityState === "visible";
      if (visible && !reduceMotion) {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(tick);
      } else {
        cancelAnimationFrame(raf);
      }
    };

    resize();
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", onVisibility);

    if (!reduceMotion && visible) {
      raf = requestAnimationFrame(tick);
    }

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  );
}
