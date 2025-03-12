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
      <div className="container mx-auto flex h-16 items-center px-4">
        <Link to="/" className="flex items-center gap-2 font-bold text-xl">
          <Heart className="h-6 w-6 text-red-500" />
          <span>Heart Disease Predictor</span>
        </Link>
        <nav className="ml-auto flex gap-6">
          <Link 
            to="/" 
            className={`text-sm font-medium transition-colors hover:text-primary ${
              isActive('/') ? 'text-primary' : 'text-muted-foreground'
            }`}
          >
            Home
          </Link>
          <Link 
            to="/predict" 
            className={`text-sm font-medium transition-colors hover:text-primary ${
              isActive('/predict') ? 'text-primary' : 'text-muted-foreground'
            }`}
          >
            Make Prediction
          </Link>
          <Link 
            to="/metrics" 
            className={`text-sm font-medium transition-colors hover:text-primary ${
              isActive('/metrics') ? 'text-primary' : 'text-muted-foreground'
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
