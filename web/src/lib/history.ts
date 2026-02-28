export interface HistoryItem {
  id: string;
  timestamp: number;
  thumbnail: string; // base64 data URL
  sfen: string;
  csa: string;
  board: string[][];
}

const STORAGE_KEY = 'sfenizer-history';
const MAX_ITEMS = 50;

export function getHistory(): HistoryItem[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

export function addToHistory(item: Omit<HistoryItem, 'id'>): void {
  const history = getHistory();
  const newItem: HistoryItem = { ...item, id: `${Date.now()}-${Math.random()}` };
  history.unshift(newItem);
  if (history.length > MAX_ITEMS) history.pop();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

export function removeFromHistory(id: string): void {
  const history = getHistory().filter((item) => item.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

export function clearHistory(): void {
  localStorage.removeItem(STORAGE_KEY);
}
