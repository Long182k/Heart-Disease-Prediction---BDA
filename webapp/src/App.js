import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./components/Home";
import PredictionForm from "./components/PredictionForm";
import Results from "./components/Results";
import ModelMetrics from "./components/ModelMetrics";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container px-4 py-8 mx-auto">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<PredictionForm />} />
            <Route path="/results" element={<Results />} />
            <Route path="/metrics" element={<ModelMetrics />} />
          </Routes>
        </main>
        <footer className="py-6 border-t">
          <div className="container mx-auto text-sm text-center text-muted-foreground">
            <p>
              {" "}
              {new Date().getFullYear()} CardioGuardian - A Big Data Analytics
              Project
            </p>
            <p className="mt-2">
              Powered by trained model by{" "}
              <span className="font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 animate-pulse">
                Long Nguyen
              </span>
            </p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
