import { apiFetch } from './api';

export interface HistoryItem {
  id: string;
  timestamp: number;
  thumbnail: string; // base64 data URL
  sfen: string;
  csa: string;
  board: string[][];
}

export async function getHistory(): Promise<HistoryItem[]> {
  const response = await apiFetch('/history');
  if (!response.ok) {
    throw new Error('Failed to load history');
  }

  const data = await response.json();
  return data.items as HistoryItem[];
}

export async function addToHistory(item: Omit<HistoryItem, 'id'>): Promise<HistoryItem> {
  const response = await apiFetch('/history', {
    method: 'POST',
    body: JSON.stringify(item)
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(errorData?.detail || 'Failed to save history');
  }

  return (await response.json()) as HistoryItem;
}

export async function removeFromHistory(id: string): Promise<void> {
  const response = await apiFetch(`/history/${id}`, {
    method: 'DELETE'
  });

  if (!response.ok) {
    throw new Error('Failed to remove history item');
  }
}

export async function clearHistory(): Promise<void> {
  const response = await apiFetch('/history', {
    method: 'DELETE'
  });

  if (!response.ok) {
    throw new Error('Failed to clear history');
  }
}
