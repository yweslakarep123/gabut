export const easeOutExpo: [number, number, number, number] = [0.16, 1, 0.3, 1];

export const revealTransition = {
  duration: 0.5,
  ease: easeOutExpo,
} as const;

export const revealInitial = {
  opacity: 0,
  y: 12,
} as const;

export const revealAnimate = {
  opacity: 1,
  y: 0,
} as const;
