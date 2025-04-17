import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import apiService from "../services/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Label } from "./ui/label";
import { AlertCircle } from "lucide-react";

const PredictionForm = () => {
  const [formData, setFormData] = useState({});
  const [validationErrors, setValidationErrors] = useState({});
  const navigate = useNavigate();

  // Fetch feature definitions using React Query
  const {
    data: features,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["features"],
    queryFn: apiService.getFeatureDefinitions,
    staleTime: Infinity, // Feature definitions won't change during the session
  });

  // Transform the features array into structured feature definitions
  const featureDefinitions = React.useMemo(() => {
    if (!features || !Array.isArray(features)) return [];

    const featureConfig = {
      age: {
        display_name: "Age (years)",
        type: "numeric",
        required: true,
        min: 18,
        max: 85,
        description: "Your age in years",
      },
      gender: {
        display_name: "Gender",
        type: "categorical",
        required: true,
        options: [
          { value: 1, label: "Female" },
          { value: 2, label: "Male" },
        ],
      },
      height: {
        display_name: "Height (cm)",
        type: "numeric",
        required: true,
        min: 140,
        max: 220,
        description: "Height in centimeters",
      },
      weight: {
        display_name: "Weight (kg)",
        type: "numeric",
        required: true,
        min: 40,
        max: 200,
        description: "Weight in kilograms",
      },
      ap_hi: {
        display_name: "Systolic Blood Pressure",
        type: "numeric",
        required: true,
        min: 80,
        max: 220,
        description: "Systolic blood pressure (upper number)",
      },
      ap_lo: {
        display_name: "Diastolic Blood Pressure",
        type: "numeric",
        required: true,
        min: 40,
        max: 140,
        description: "Diastolic blood pressure (lower number)",
      },
      cholesterol: {
        display_name: "Cholesterol",
        type: "categorical",
        required: true,
        options: [
          { value: 1, label: "Normal" },
          { value: 2, label: "Above Normal" },
          { value: 3, label: "Well Above Normal" },
        ],
      },
      gluc: {
        display_name: "Glucose",
        type: "categorical",
        required: true,
        options: [
          { value: 1, label: "Normal" },
          { value: 2, label: "Above Normal" },
          { value: 3, label: "Well Above Normal" },
        ],
      },
      smoke: {
        display_name: "Smoker",
        type: "categorical",
        required: true,
        options: [
          { value: 0, label: "No" },
          { value: 1, label: "Yes" },
        ],
      },
      alco: {
        display_name: "Alcohol Consumption",
        type: "categorical",
        required: true,
        options: [
          { value: 0, label: "No" },
          { value: 1, label: "Yes" },
        ],
      },
      active: {
        display_name: "Physically Active",
        type: "categorical",
        required: true,
        options: [
          { value: 0, label: "No" },
          { value: 1, label: "Yes" },
        ],
      },
      bp_category_encoded: {
        display_name: "Blood Pressure Category",
        type: "categorical",
        required: true,
        options: [
          { value: "Normal", label: "Normal" },
          { value: "Elevated", label: "Elevated" },
          { value: "Hypertension Stage 1", label: "Hypertension Stage 1" },
          { value: "Hypertension Stage 2", label: "Hypertension Stage 2" },
        ],
        description: "Your blood pressure category",
      },
    };

    // Filter to only include features that are in the API response
    // and exclude derived features that shouldn't be in the form
    return features
      .filter((name) =>
        [
          "age",
          "gender",
          "height",
          "weight",
          "ap_hi",
          "ap_lo",
          "cholesterol",
          "gluc",
          "smoke",
          "alco",
          "active",
          "bp_category_encoded",
        ].includes(name)
      )
      .map((name) => ({
        name,
        ...featureConfig[name],
      }));
  }, [features]);
  console.log("🚀 featureDefinitions:", featureDefinitions);

  // Mutation for making predictions
  const predictionMutation = useMutation({
    mutationFn: (data) => {
      // Convert age from years to days (multiply by 365.25 to account for leap years)
      const ageInDays = Math.round(parseFloat(data.age) * 365.25);

      // Calculate BMI if needed
      const bmi = (
        parseFloat(data.weight) /
        (parseFloat(data.height) / 100) ** 2
      ).toFixed(1);

      // Use the selected blood pressure category instead of calculating it
      const bp_category_encoded = data.bp_category_encoded || "Unknown";

      // Ensure all required fields are present and convert to correct types
      const processedData = {
        age_years: parseInt(data.age), // Send age in years to the API
        gender: parseInt(data.gender),
        height: parseInt(data.height),
        weight: parseFloat(data.weight),
        ap_hi: parseInt(data.ap_hi),
        ap_lo: parseInt(data.ap_lo),
        cholesterol: parseInt(data.cholesterol),
        gluc: parseInt(data.gluc),
        smoke: parseInt(data.smoke),
        alco: parseInt(data.alco),
        active: parseInt(data.active),
        bmi: parseFloat(bmi),
        bp_category_encoded: bp_category_encoded,
      };
      return apiService.makePrediction(processedData);
    },
    onSuccess: (data) => {
      // Navigate to results page with prediction data
      navigate("/results", {
        state: {
          result: data,
          patientData: formData,
        },
      });
    },
    onError: (error) => {
      setValidationErrors({
        form: error.message || "Failed to make prediction. Please try again.",
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

    if (!featureDefinitions || featureDefinitions.length === 0) return errors;

    // Required fields based on API requirements
    const requiredFields = [
      "age",
      "gender",
      "height",
      "weight",
      "ap_hi",
      "ap_lo",
      "cholesterol",
      "gluc",
      "smoke",
      "alco",
      "active",
      "bp_category_encoded",
    ];

    // Check if all required fields are present
    requiredFields.forEach((field) => {
      if (!formData[field] && formData[field] !== 0) {
        errors[field] = "This field is required";
      }
    });

    featureDefinitions.forEach((feature) => {
      const value = formData[feature.name];

      // Check if required field is empty
      if (feature.required && (value === undefined || value === "")) {
        errors[feature.name] = "This field is required";
      }

      // Check if number is within bounds
      if (value !== undefined && value !== "" && feature.type === "numeric") {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
          errors[feature.name] = "Must be a number";
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

    if (feature.type === "categorical" && feature.options) {
      return (
        <div className="grid gap-2" key={feature.name}>
          <Label htmlFor={feature.name}>{feature.display_name}</Label>
          <Select
            value={formData[feature.name] || ""}
            onValueChange={(value) => handleInputChange(feature.name, value)}
          >
            <SelectTrigger className={errorMessage ? "border-destructive" : ""}>
              <SelectValue
                placeholder={`Select ${feature.display_name.toLowerCase()}`}
              />
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
            type={feature.type === "numeric" ? "number" : "text"}
            placeholder={feature.description || feature.display_name}
            value={formData[feature.name] || ""}
            onChange={(e) => handleInputChange(feature.name, e.target.value)}
            className={errorMessage ? "border-destructive" : ""}
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
          Unable to load prediction form. Please refresh the page or try again
          later.
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
            {featureDefinitions &&
              featureDefinitions.map((feature) => renderInputField(feature))}
            {validationErrors.form && (
              <div className="rounded-lg border-destructive/50 bg-destructive/5 p-4">
                <div className="flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-destructive" />
                  <p className="text-sm text-destructive">
                    {validationErrors.form}
                  </p>
                </div>
              </div>
            )}
          </CardContent>
          <CardFooter className="flex justify-between">
            <Button
              type="button"
              variant="outline"
              onClick={() => setFormData({})}
            >
              Reset
            </Button>
            <Button type="submit" disabled={predictionMutation.isPending}>
              {predictionMutation.isPending ? (
                <>
                  <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-solid border-current border-r-transparent"></span>
                  Processing...
                </>
              ) : (
                "Get Prediction"
              )}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
};

export default PredictionForm;
