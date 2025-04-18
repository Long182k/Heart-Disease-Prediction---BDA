import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { AlertTriangle, CheckCircle2, Info, Heart, Activity } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import apiService from '../services/api';

const Results = () => {
  const location = useLocation();
  const { predictionId } = useParams();
  const { result: locationResult, patientData: locationPatientData } = location.state || {};
  
  // If we have a predictionId, fetch the prediction
  const { data: fetchedPrediction, isLoading } = useQuery({
    queryKey: ['prediction', predictionId],
    queryFn: () => apiService.getPredictionById(predictionId),
    enabled: !!predictionId,
  });
  
  // Use either the fetched prediction or the one from location state
  const result = predictionId && fetchedPrediction 
    ? fetchedPrediction.prediction_result 
    : locationResult;
    
  const patientData = predictionId && fetchedPrediction 
    ? fetchedPrediction.patient_data 
    : locationPatientData;

  // If loading a prediction by ID, show loading state
  if (predictionId && isLoading) {
    return (
      <div className="flex flex-col justify-center items-center pt-12 space-y-4">
        <h3 className="text-xl font-semibold">Loading prediction...</h3>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex flex-col justify-center items-center pt-12 space-y-4">
        <h3 className="text-xl font-semibold">No prediction data available</h3>
        <p className="text-muted-foreground">Please complete the prediction form to see results</p>
        <Link to="/predict">
          <Button>Go to Prediction Form</Button>
        </Link>
      </div>
    );
  }
  
  const getRiskLevelIcon = () => {
    switch(result.risk_level) {
      case 'Low':
        return <CheckCircle2 className="w-12 h-12 text-green-500" />;
      case 'Moderate':
        return <Info className="w-12 h-12 text-amber-500" />;
      case 'High':
      case 'Very High':
        return <AlertTriangle className="w-12 h-12 text-red-500" />;
      default:
        return <Info className="w-12 h-12 text-blue-500" />;
    }
  };

  const getRiskClass = () => {
    switch(result.risk_level) {
      case 'Low':
        return 'bg-green-50 border-green-200';
      case 'Moderate':
        return 'bg-amber-50 border-amber-200';
      case 'High':
      case 'Very High':
        return 'bg-red-50 border-red-200';
      default:
        return '';
    }
  };

  const getCardioStatusClass = () => {
    return result.cardio === 1 
      ? 'bg-red-100 text-red-800 border-red-300' 
      : 'bg-green-100 text-green-800 border-green-300';
  };

  const getCardioStatusText = () => {
    return result.cardio === 1 
      ? 'Present' 
      : 'Absent';
  };

  const getRiskMessage = () => {
    switch(result.risk_level) {
      case 'Low':
        return 'Your risk of heart disease appears to be low based on the provided information.';
      case 'Moderate':
        return 'You have a moderate risk of heart disease. Consider consulting with a healthcare provider.';
      case 'High':
        return 'Your risk factors indicate a high likelihood of heart disease. We strongly recommend consulting with a healthcare provider.';
      case 'Very High':
        return 'Your risk factors indicate a very high likelihood of heart disease. We strongly recommend consulting with a healthcare provider immediately.';
      default:
        return 'Please consult with a healthcare provider to interpret these results.';
    }
  };

  // Format patient data for display
  const formatPatientData = () => {
    if (!patientData) return {};
    
    const displayLabels = {
      age: 'Age (days)',
      age_years: 'Age (years)',
      gender: 'Gender',
      height: 'Height (cm)',
      weight: 'Weight (kg)',
      ap_hi: 'Systolic BP',
      ap_lo: 'Diastolic BP',
      cholesterol: 'Cholesterol',
      gluc: 'Glucose',
      smoke: 'Smoker',
      alco: 'Alcohol Consumption',
      active: 'Physically Active',
      bmi: 'BMI',
      bp_category: 'BP Category'
    };
    
    const formattedData = {};
    
    Object.entries(patientData).forEach(([key, value]) => {
      const label = displayLabels[key] || key;
      
      // Format boolean values
      if (key === 'smoke' || key === 'alco' || key === 'active') {
        formattedData[label] = value === '1' || value === 1 ? 'Yes' : 'No';
      }
      // Format gender
      else if (key === 'gender') {
        formattedData[label] = value === '1' || value === 1 ? 'Female' : 'Male';
      }
      // Format cholesterol and glucose levels
      else if (key === 'cholesterol') {
        const levels = ['Normal', 'Above Normal', 'Well Above Normal'];
        formattedData[label] = levels[parseInt(value) - 1] || value;
      }
      else if (key === 'gluc') {
        const levels = ['Normal', 'Above Normal', 'Well Above Normal'];
        formattedData[label] = levels[parseInt(value) - 1] || value;
      }
      else {
        formattedData[label] = value;
      }
    });

    // Add BMI from result input_summary if available
    if (result.input_summary && result.input_summary.bmi) {
      formattedData['BMI'] = result.input_summary.bmi.toFixed(1);
    }
    
    return formattedData;
  };

  const formattedPatientData = formatPatientData();

  return (
    <div className="space-y-8">
      <div className="space-y-2 text-center">
        <h1 className="text-3xl font-bold">Cardiovascular Disease Prediction</h1>
        <p className="text-muted-foreground">
          Assessment results based on your health information
        </p>
      </div>

      {/* Primary Result Card */}
      <Card className={`mx-auto max-w-2xl border-2 ${getRiskClass()}`}>
        <CardHeader className="pb-2 text-center">
          <div className="flex justify-center mb-2">
            {getRiskLevelIcon()}
          </div>
          <div className="flex gap-2 justify-center items-center mb-2">
            <Heart className={`h-5 w-5 ${result.cardio === 1 ? 'text-red-500' : 'text-green-500'}`} />
            <CardTitle className="text-2xl">Cardiovascular Disease Assessment</CardTitle>
          </div>
          <CardDescription>
            Based on your provided health information
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center">
          {/* Cardio Status Badge */}
          <div className="mb-4">
            <Badge className={`px-3 py-1 text-sm ${getCardioStatusClass()}`}>
              Cardiovascular Disease: {getCardioStatusText()}
            </Badge>
          </div>
          
          <div className="pb-2 text-3xl font-bold">{result.risk_level} Risk</div>
          <div className="pb-4 font-medium">
            Probability: {(result.probability * 100)?.toFixed(2)}%
          </div>
          <p className="text-muted-foreground">
            {getRiskMessage()}
          </p>
          {result.reason && (
            <div className="mt-4 text-sm text-muted-foreground">
              <strong>Reason:</strong> {result.reason}
            </div>
          )}
          {result.advice && (
            <div className="mt-2 text-sm text-muted-foreground">
              <strong>Advice:</strong> {result.advice}
            </div>
          )}
        </CardContent>
        <CardFooter className="flex flex-col gap-2">
          <div className="text-xs text-center text-muted-foreground">
            This prediction is based on machine learning models and is provided for informational purposes only.
          </div>
          {result.model_type && (
            <div className="text-xs text-center text-muted-foreground">
              Model: <span className="font-medium">{result.model_type.toUpperCase()}</span>
            </div>
          )}
        </CardFooter>
      </Card>

      {/* Risk Factors Card */}
      <Card className="mx-auto max-w-3xl">
        <CardHeader>
          <div className="flex gap-2 items-center">
            <Activity className="w-5 h-5 text-blue-500" />
            <CardTitle>Risk Factors</CardTitle>
          </div>
          <CardDescription>
            Key health indicators used for this prediction
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {Object.entries(formattedPatientData).map(([key, value]) => (
              <div key={key} className="flex justify-between pb-2 border-b">
                <span className="font-medium">{key}:</span>
                <span className="text-muted-foreground">{value}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="flex gap-4 justify-center pt-4">
        <Link to="/predict">
          <Button>New Prediction</Button>
        </Link>
        <Link to="/metrics">
          <Button variant="outline">View Model Metrics</Button>
        </Link>
      </div>

      <div className="p-4 mx-auto max-w-3xl text-sm text-center rounded-md text-muted-foreground bg-accent/50">
        <p>
          <strong>Disclaimer:</strong> This prediction is based on machine learning models and is for 
          educational purposes only. It should not replace professional medical advice. Please consult 
          with a healthcare provider for proper diagnosis and treatment.
        </p>
      </div>
    </div>
  );
};

export default Results;
