import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import apiService from '../services/api';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Label } from './ui/label';
import { AlertCircle } from 'lucide-react';

const PredictionForm = () => {
  const [formData, setFormData] = useState({});
  const [validationErrors, setValidationErrors] = useState({});
  const navigate = useNavigate();

  // Fetch feature definitions using React Query
  const { data: features, isLoading, error } = useQuery({
    queryKey: ['features'],
    queryFn: apiService.getFeatureDefinitions,
    staleTime: Infinity, // Feature definitions won't change during the session
  });

  // Mutation for making predictions
  const predictionMutation = useMutation({
    mutationFn: (data) => apiService.makePrediction(data),
    onSuccess: (data) => {
      // Navigate to results page with prediction data
      navigate('/results', { 
        state: { 
          result: data, 
          patientData: formData 
        } 
      });
    },
  });

  // Handle input change for all form fields
  const handleInputChange = (name, value) => {
    setFormData({
      ...formData,
      [name]: value,
    });
    
    // Clear validation error for this field if it exists
    if (validationErrors[name]) {
      const newErrors = { ...validationErrors };
      delete newErrors[name];
      setValidationErrors(newErrors);
    }
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Validate form data
    const errors = validateFormData();
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      return;
    }
    
    // Submit data for prediction
    predictionMutation.mutate(formData);
  };

  // Validate all form fields
  const validateFormData = () => {
    const errors = {};
    
    if (!features) return errors;
    
    features.forEach(feature => {
      const value = formData[feature.name];
      
      // Check if required field is empty
      if (feature.required && (value === undefined || value === '')) {
        errors[feature.name] = 'This field is required';
      }
      
      // Check if number is within bounds
      if (value !== undefined && value !== '' && feature.type === 'numeric') {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
          errors[feature.name] = 'Must be a number';
        } else if (feature.min !== undefined && numValue < feature.min) {
          errors[feature.name] = `Minimum value is ${feature.min}`;
        } else if (feature.max !== undefined && numValue > feature.max) {
          errors[feature.name] = `Maximum value is ${feature.max}`;
        }
      }
    });
    
    return errors;
  };

  // Render appropriate input field based on feature type
  const renderInputField = (feature) => {
    const errorMessage = validationErrors[feature.name];
    
    if (feature.type === 'categorical' && feature.options) {
      return (
        <div className="grid gap-2" key={feature.name}>
          <Label htmlFor={feature.name}>{feature.display_name}</Label>
          <Select
            value={formData[feature.name] || ''}
            onValueChange={(value) => handleInputChange(feature.name, value)}
          >
            <SelectTrigger className={errorMessage ? 'border-destructive' : ''}>
              <SelectValue placeholder={`Select ${feature.display_name.toLowerCase()}`} />
            </SelectTrigger>
            <SelectContent>
              {feature.options.map((option) => (
                <SelectItem key={option.value} value={option.value.toString()}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {errorMessage && (
            <p className="text-sm text-destructive">{errorMessage}</p>
          )}
        </div>
      );
    } else {
      return (
        <div className="grid gap-2" key={feature.name}>
          <Label htmlFor={feature.name}>{feature.display_name}</Label>
          <Input
            id={feature.name}
            type={feature.type === 'numeric' ? 'number' : 'text'}
            placeholder={feature.description || feature.display_name}
            value={formData[feature.name] || ''}
            onChange={(e) => handleInputChange(feature.name, e.target.value)}
            className={errorMessage ? 'border-destructive' : ''}
            min={feature.min}
            max={feature.max}
            step={feature.step || 1}
          />
          {errorMessage && (
            <p className="text-sm text-destructive">{errorMessage}</p>
          )}
        </div>
      );
    }
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
          <h3 className="font-medium">Failed to load form</h3>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Unable to load prediction form. Please refresh the page or try again later.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="space-y-2 text-center">
        <h1 className="text-3xl font-bold">Heart Disease Risk Assessment</h1>
        <p className="text-muted-foreground">
          Enter your health information below to receive a prediction
        </p>
      </div>

      <Card className="mx-auto max-w-3xl">
        <CardHeader>
          <CardTitle>Patient Information</CardTitle>
          <CardDescription>
            Please provide accurate information for a more reliable prediction.
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent className="grid gap-6">
            {features && features.map((feature) => renderInputField(feature))}
          </CardContent>
          <CardFooter className="flex justify-between">
            <Button
              type="button"
              variant="outline"
              onClick={() => setFormData({})}
            >
              Reset
            </Button>
            <Button 
              type="submit"
              disabled={predictionMutation.isPending}
            >
              {predictionMutation.isPending ? (
                <>
                  <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-solid border-current border-r-transparent">
                  </span>
                  Processing...
                </>
              ) : "Get Prediction"}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
};

export default PredictionForm;
