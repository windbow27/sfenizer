const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? (import.meta.env.PROD ? '/api' : 'http://localhost:8000');

export function getApiBaseUrl() {
  return API_BASE_URL;
}

export function getStoredToken() {
  return localStorage.getItem('sfenizer-token');
}

export async function apiFetch(path: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers || {});
  const token = getStoredToken();

  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  if (API_BASE_URL.includes('ngrok')) {
    headers.set('ngrok-skip-browser-warning', 'true');
  }

  return fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers
  });
}
