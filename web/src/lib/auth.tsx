import React, { createContext, useContext, useEffect, useState } from 'react';
import { apiFetch, getStoredToken } from './api';

export interface AuthUser {
  id: number;
  username: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);
const TOKEN_KEY = 'sfenizer-token';

async function authenticate(
  path: '/auth/login' | '/auth/register',
  username: string,
  password: string
) {
  const response = await apiFetch(path, {
    method: 'POST',
    body: JSON.stringify({ username, password })
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(errorData?.detail || 'Authentication failed');
  }

  const data = await response.json();
  localStorage.setItem(TOKEN_KEY, data.token);
  return data.user as AuthUser;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      setIsLoading(false);
      return;
    }

    let active = true;
    apiFetch('/auth/me')
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Session expired');
        }
        const data = await response.json();
        if (active) {
          setUser(data.user as AuthUser);
        }
      })
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY);
        if (active) {
          setUser(null);
        }
      })
      .finally(() => {
        if (active) {
          setIsLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const login = async (username: string, password: string) => {
    const nextUser = await authenticate('/auth/login', username, password);
    setUser(nextUser);
  };

  const register = async (username: string, password: string) => {
    const nextUser = await authenticate('/auth/register', username, password);
    setUser(nextUser);
  };

  const logout = async () => {
    try {
      await apiFetch('/auth/logout', { method: 'POST' });
    } finally {
      localStorage.removeItem(TOKEN_KEY);
      setUser(null);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: Boolean(user),
        login,
        register,
        logout
      }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
