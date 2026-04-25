import React from 'react';
import { Routes, Route } from 'react-router-dom';
import NavBar from './components/NavBar';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import ImageSfenizer from './pages/ImageSfenizer';
import VideoSfenizer from './pages/VideoSfenizer';
import History from './pages/History';
import Login from './pages/Login';

const App: React.FC = () => {
  return (
    <div className='min-h-screen bg-background'>
      <NavBar />
      <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
        <div className='flex space-x-16'>
          <Sidebar />
          <main className='flex-1 min-w-0'>
            <Routes>
              <Route path='/' element={<Home />} />
              <Route path='/login' element={<Login />} />
              <Route path='/image' element={<ImageSfenizer />} />
              <Route path='/video' element={<VideoSfenizer />} />
              <Route path='/history' element={<History />} />
              <Route path='*' element={<Home />} />
            </Routes>
          </main>
        </div>
      </div>
    </div>
  );
};

export default App;
