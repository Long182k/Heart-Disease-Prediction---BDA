import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Heart } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();
  
  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <header className="border-b">
      <div className="container flex items-center px-4 mx-auto h-16">
        <Link to="/" className="flex gap-2 items-center text-xl font-bold">
          <Heart className="w-6 h-6 text-red-500" />
          <span className="font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 animate-pulse">CardioGuardian</span>
        </Link>
        <nav className="flex gap-6 ml-auto">
          <Link 
            to="/" 
            className={`text-sm font-medium transition-colors hover:text-primary ${ isActive('/') ? 'text-primary' : 'text-muted-foreground'
            }`}
          >
            Home
          </Link>
          <Link 
            to="/predict" 
            className={`text-sm font-medium transition-colors hover:text-primary ${ isActive('/predict') ? 'text-primary' : 'text-muted-foreground'
            }`}
          >
            Make Prediction
          </Link>
          <Link 
            to="/metrics" 
            className={`text-sm font-medium transition-colors hover:text-primary ${ isActive('/metrics') ? 'text-primary' : 'text-muted-foreground'
            }`}
          >
            Model Metrics
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
