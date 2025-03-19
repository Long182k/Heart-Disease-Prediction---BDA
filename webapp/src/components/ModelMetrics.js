import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { AlertCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import apiService from '../services/api';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ModelMetrics = () => {
  // Fetch model metrics using React Query
  const { data: metrics, isLoading, error } = useQuery({
    queryKey: ['modelMetrics'],
    queryFn: apiService.getModelMetrics,
    staleTime: Infinity, // Metrics won't change during the session
  });


  // Chart options
  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Performance Comparison',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
      },
    },
  };

  // Time chart options
  const timeOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Training Times (seconds)',
      },
    },
  };

  // Prepare performance chart data
  const prepareChartData = () => {
    const labels = metrics.map(m => m.model_name.replace('_', ' ').toUpperCase());
    
    return {
      labels,
      datasets: [
        {
          label: 'Accuracy',
          data: metrics.map(m => m.accuracy),
          backgroundColor: 'rgba(54, 162, 235, 0.6)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1,
        },
        {
          label: 'AUC',
          data: metrics.map(m => m.auc),
          backgroundColor: 'rgba(153, 102, 255, 0.6)',
          borderColor: 'rgba(153, 102, 255, 1)',
          borderWidth: 1,
        },
        {
          label: 'F1 Score',
          data: metrics.map(m => m.f1),
          backgroundColor: 'rgba(255, 159, 64, 0.6)',
          borderColor: 'rgba(255, 159, 64, 1)',
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare time chart data
  const prepareTimeChartData = () => {
    const labels = metrics.map(m => m.model_name.replace('_', ' ').toUpperCase());
    
    return {
      labels,
      datasets: [
        {
          label: 'Training Time (seconds)',
          data: metrics.map(m => m.training_time),
          backgroundColor: 'rgba(255, 99, 132, 0.6)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 1,
        },
      ],
    };
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border-destructive/50 bg-destructive/5 p-6">
        <div className="flex items-center gap-2">
          <AlertCircle className="h-5 w-5 text-destructive" />
          <h3 className="font-medium">Failed to load model metrics</h3>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Please refresh the page or try again later.
        </p>
      </div>
    );
  }

  if (!metrics || metrics.length === 0) {
    return (
      <div className="text-center p-8">
        <h2 className="text-xl font-semibold">No model metrics available</h2>
        <p className="text-muted-foreground mt-2">
          Model performance data has not been generated yet.
        </p>
      </div>
    );
  }

  // Find best model based on AUC score
  const bestModel = metrics.reduce((prev, current) => 
    (prev.auc > current.auc) ? prev : current
  );

  return (
    <div className="space-y-8">
      <div className="space-y-2 text-center">
        <h1 className="text-3xl font-bold">Model Performance Metrics</h1>
        <p className="text-muted-foreground">
          Comparing the performance of different machine learning models for heart disease prediction
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
            <CardDescription>
              Comparing accuracy, AUC, and F1 scores across models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Bar data={prepareChartData()} options={options} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Training Time</CardTitle>
            <CardDescription>
              Time taken to train each model (in seconds)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Bar data={prepareTimeChartData()} options={timeOptions} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Detailed Model Metrics</CardTitle>
          <CardDescription>
            Comprehensive performance metrics for each model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b">
                  <th className="py-3 px-4 text-left font-medium">Model</th>
                  <th className="py-3 px-4 text-left font-medium">Accuracy</th>
                  <th className="py-3 px-4 text-left font-medium">Precision</th>
                  <th className="py-3 px-4 text-left font-medium">Recall</th>
                  <th className="py-3 px-4 text-left font-medium">F1 Score</th>
                  <th className="py-3 px-4 text-left font-medium">AUC</th>
                  <th className="py-3 px-4 text-left font-medium">Training Time (s)</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((model) => (
                  <tr key={model.model_name} className="border-b hover:bg-muted/50">
                    <td className="py-3 px-4 font-medium">{model.model_name.replace('_', ' ').toUpperCase()}</td>
                    <td className="py-3 px-4">{model.accuracy?.toFixed(4)}</td>
                    <td className="py-3 px-4">{model.precision?.toFixed(4)}</td>
                    <td className="py-3 px-4">{model.recall?.toFixed(4)}</td>
                    <td className="py-3 px-4">{model.f1?.toFixed(4)}</td>
                    <td className="py-3 px-4">{model.auc?.toFixed(4)}</td>
                    <td className="py-3 px-4">{model.training_time?.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-primary/5 border-primary/20">
        <CardHeader>
          <CardTitle>Best Performing Model</CardTitle>
        </CardHeader>
        <CardContent>
          <p>
            The <span className="font-bold">{bestModel.model_name.replace('_', ' ').toUpperCase()}</span> model 
            achieved the highest AUC score of <span className="font-bold">{bestModel.auc?.toFixed(4)}</span>.
          </p>
          <p className="mt-2 text-sm text-muted-foreground">
            AUC (Area Under the ROC Curve) is a key metric for evaluating classification model performance,
            especially for imbalanced datasets like those commonly found in medical applications.
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelMetrics;
