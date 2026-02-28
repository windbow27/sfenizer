import { useEffect, useRef, useState } from 'react';

/**
 * Hook that returns true once the component has mounted, creating a
 * one-shot entrance trigger for CSS transition classes.
 * @param delay  ms to wait before flipping to `true` (stagger support)
 */
export function useEnter(delay = 0): boolean {
  const [entered, setEntered] = useState(false);
  useEffect(() => {
    const id = setTimeout(() => setEntered(true), delay);
    return () => clearTimeout(id);
  }, [delay]);
  return entered;
}

/**
 * IntersectionObserver-backed hook.  Returns [ref, isVisible].
 * Element fades/slides in once it scrolls into view.
 */
export function useInView<T extends HTMLElement = HTMLDivElement>(
  options?: IntersectionObserverInit
): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { threshold: 0.1, ...options }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [options]);

  return [ref, visible];
}

/** Shared transition class builder */
export const fadeUp = (visible: boolean, delay = 0) =>
  `transition-all duration-500 ease-out ${
    visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
  }` + (delay ? ` delay-[${delay}ms]` : '');

/** Shared scale-in class builder */
export const scaleIn = (visible: boolean) =>
  `transition-all duration-400 ease-out ${
    visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
  }`;
