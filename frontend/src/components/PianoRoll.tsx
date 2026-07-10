"use client";

import { useEffect, useRef } from "react";

export interface RollNote {
  time: number; // seconds from the start of the piece
  duration: number; // seconds
  midi: number; // 0-127
  velocity: number; // 0-1
}

interface PianoRollProps {
  notes: RollNote[];
  /** Seconds elapsed. Drives the playhead; pass a ref-like getter to avoid re-rendering. */
  getPlayhead: () => number;
  duration: number;
  height?: number;
  className?: string;
}

const BRASS = "oklch(74% 0.1 80)";
const BRASS_DIM = "oklch(74% 0.1 80 / 45%)";
const VERDIGRIS = "oklch(64% 0.09 175)";
const GRID = "oklch(38% 0.02 260 / 30%)";

// A piano's black keys, as pitch classes. Shading their rows gives the eye an
// octave reference without drawing a keyboard down the side.
const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);

/**
 * Canvas piano roll: x is time, y is pitch, each note is a bar.
 *
 * Drawn on a canvas rather than as SVG/DOM nodes because a 300-note piece
 * repainted every animation frame for the playhead would be 300 style
 * recalculations per frame. The note layer is painted once into an offscreen
 * canvas and blitted each frame; only the playhead is actually redrawn.
 */
export function PianoRoll({ notes, getPlayhead, duration, height = 180, className = "" }: PianoRollProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layerRef = useRef<HTMLCanvasElement | null>(null);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || notes.length === 0) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;

    canvas.width = width * dpr;
    canvas.height = height * dpr;

    const lowest = Math.min(...notes.map((n) => n.midi));
    const highest = Math.max(...notes.map((n) => n.midi));
    // Pad the pitch range so the outermost notes are not flush against the edge.
    const low = lowest - 2;
    const span = Math.max(highest - lowest + 4, 12);

    const totalTime = duration || Math.max(...notes.map((n) => n.time + n.duration));
    const xOf = (seconds: number) => (seconds / totalTime) * width;
    const yOf = (midi: number) => height - ((midi - low) / span) * height;
    const rowHeight = Math.max(height / span, 2);

    const layer = document.createElement("canvas");
    layer.width = canvas.width;
    layer.height = canvas.height;
    const layerContext = layer.getContext("2d");
    if (!layerContext) return;
    layerContext.scale(dpr, dpr);

    for (let midi = low; midi < low + span; midi += 1) {
      if (!BLACK_KEYS.has(((midi % 12) + 12) % 12)) continue;
      layerContext.fillStyle = GRID;
      layerContext.fillRect(0, yOf(midi) - rowHeight, width, rowHeight);
    }

    for (const note of notes) {
      const x = xOf(note.time);
      const barWidth = Math.max(xOf(note.time + note.duration) - x, 1.5);
      layerContext.fillStyle = note.velocity > 0.7 ? BRASS : BRASS_DIM;
      layerContext.fillRect(x, yOf(note.midi) - rowHeight, barWidth, rowHeight);
    }

    layerRef.current = layer;

    const draw = () => {
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(layer, 0, 0);
      context.scale(dpr, dpr);

      const elapsed = getPlayhead();
      if (elapsed > 0 && elapsed <= totalTime) {
        const x = xOf(elapsed);
        context.strokeStyle = VERDIGRIS;
        context.lineWidth = 1.5;
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
      }

      frameRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(frameRef.current);
  }, [notes, duration, height, getPlayhead]);

  if (notes.length === 0) return null;

  return (
    <canvas
      ref={canvasRef}
      style={{ height, width: "100%" }}
      className={`block ${className}`}
      role="img"
      aria-label={`Piano roll of ${notes.length} notes spanning ${duration.toFixed(1)} seconds`}
    />
  );
}
