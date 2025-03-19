import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { AlertTriangle, CheckCircle2, Info } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';

const Results = () => {
  const location = useLocation();
  const { result, patientData } = location.state || {};

  // If no result data is available, redirect to prediction form
  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center space-y-4 pt-12">
        <h3 className="text-xl font-semibold">No prediction data available</h3>
        <p className="text-muted-foreground">Please complete the prediction form to see results</p>
        <Link to="/predict">
          <Button>Go to Prediction Form</Button>
        </Link>
      </div>
    );
  }

  // Determine risk level icon and color
  const getRiskLevelIcon = () => {
    switch(result.risk_level) {
      case 'Low':
        return <CheckCircle2 className="h-12 w-12 text-green-500" />;
      case 'Moderate':
        return <Info className="h-12 w-12 text-amber-500" />;
      case 'High':
      case 'Very High':
        return <AlertTriangle className="h-12 w-12 text-red-500" />;
      default:
        return <Info className="h-12 w-12 text-blue-500" />;
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
      gender: 'Gender',
      height: 'Height (cm)',
      weight: 'Weight (kg)',
      ap_hi: 'Systolic BP',
      ap_lo: 'Diastolic BP',
      cholesterol: 'Cholesterol',
      gluc: 'Glucose',
      smoke: 'Smoker',
      alco: 'Alcohol Consumption',
      active: 'Physically Active'
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
    
    return formattedData;
  };

  const formattedPatientData = formatPatientData();

  return (
    <div className="space-y-8">
      <div className="space-y-2 text-center">
        <h1 className="text-3xl font-bold">Prediction Results</h1>
        <p className="text-muted-foreground">
          Your heart disease risk assessment based on the provided information
        </p>
      </div>

      <Card className={`mx-auto max-w-2xl ${getRiskClass()}`}>
        <CardHeader className="text-center pb-2">
          <div className="flex justify-center mb-2">
            {getRiskLevelIcon()}
          </div>
          <CardTitle className="text-2xl">Heart Disease Risk Assessment</CardTitle>
          <CardDescription>
            Based on your provided health information
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center">
          <div className="text-3xl font-bold pb-2">{result.risk_level}</div>
          <div className="font-medium pb-4">
            Probability: {(result.probability * 100)?.toFixed(2)}%
          </div>
          <p className="text-muted-foreground">
            {getRiskMessage()}
          </p>
          {result.model_used && (
            <div className="mt-4 text-sm text-muted-foreground">
              Model used: <span className="font-medium">{result.model_used.toUpperCase()}</span>
            </div>
          )}
        </CardContent>
        <CardFooter className="flex justify-center">
          <div className="text-xs text-muted-foreground text-center">
            This prediction is based on machine learning models and is provided for informational purposes only.
          </div>
        </CardFooter>
      </Card>

      <Card className="mx-auto max-w-3xl">
        <CardHeader>
          <CardTitle>Patient Information</CardTitle>
          <CardDescription>
            Data used for this prediction
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.entries(formattedPatientData).map(([key, value]) => (
              <div key={key} className="flex justify-between border-b pb-2">
                <span className="font-medium">{key}:</span>
                <span className="text-muted-foreground">{value}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-center gap-4 pt-4">
        <Link to="/predict">
          <Button>New Prediction</Button>
        </Link>
        <Link to="/metrics">
          <Button variant="outline">View Model Metrics</Button>
        </Link>
      </div>

      <div className="text-center text-sm text-muted-foreground bg-accent/50 p-4 rounded-md mx-auto max-w-3xl">
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
