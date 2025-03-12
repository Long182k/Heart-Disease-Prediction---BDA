import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './components/Home';
import PredictionForm from './components/PredictionForm';
import Results from './components/Results';
import ModelMetrics from './components/ModelMetrics';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<PredictionForm />} />
            <Route path="/results" element={<Results />} />
            <Route path="/metrics" element={<ModelMetrics />} />
          </Routes>
        </main>
        <footer className="border-t py-6">
          <div className="container mx-auto text-center text-sm text-muted-foreground">
            <p> {new Date().getFullYear()} Heart Disease Prediction - A Big Data Analytics Project</p>
            <p className="mt-2">Powered by Machine Learning</p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
