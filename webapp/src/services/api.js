import axios from "axios";
import authService from "./authService";
const API_URL = "http://localhost:5002/";
const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});
// Function to set up auth token on api requests
export const setupApiAuth = () => {
  const token = authService.getToken();
  if (token) {
    api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  }
};

// Set up auth initially
setupApiAuth();

// Helper function to handle API errors
const handleApiError = (error) => {
  console.error("API Error:", error.response?.data || error.message);
  throw (
    error.response?.data || {
      message: "An error occurred with the API request",
    }
  );
};

// API service functions
const apiService = {
  // Get feature definitions for the prediction form
  getFeatureDefinitions: async () => {
    try {
      const response = await api.get("/api/features");
      console.log("Feature definitions response:", response);
      return response.data.features || response.data;
    } catch (error) {
      console.error("Error fetching feature definitions:", error);
      throw (
        error.response?.data || {
          message: "Failed to fetch feature definitions",
        }
      );
    }
  },

  // Submit data for prediction
  makePrediction: async (data) => {
    try {
      const response = await api.post("/api/predict", data);
      console.log("Prediction response:", response);
      return response.data;
    } catch (error) {
      console.error("Error making prediction:", error);
      throw error.response?.data || { message: "Failed to make prediction" };
    }
  },

  getModelMetrics: async () => {
    try {
      // Add leading slash to ensure correct URL
      const response = await api.get("/api/metrics");
      console.log("Model metrics response:", response);
      console.log("JSON.parse(response.data):", JSON.parse(response.data));

      // Handle string response if needed
      if (typeof response.data === "string") {
        console.log("Parsing string response as JSON...");
        try {
          return JSON.parse(response.data);
        } catch (parseError) {
          console.error("Error parsing metrics JSON:", parseError);
          throw new Error("Invalid response format");
        }
      }

      // Ensure we're returning an array
      if (Array.isArray(response.data)) {
        return response.data;
      } else if (response.data && Array.isArray(response.data.data)) {
        return response.data.data;
      } else {
        console.error("Unexpected response format:", response.data);
        throw new Error("Response is not an array");
      }
    } catch (error) {
      console.error("Error fetching model metrics:", error);
      throw (
        error.response?.data || { message: "Failed to fetch model metrics" }
      );
    }
  },

  // Get user predictions
  getUserPredictions: async () => {
    try {
      const response = await api.get("/api/user/predictions");
      console.log("User predictions response:", response);
      return response.data;
    } catch (error) {
      console.error("Error fetching user predictions:", error);
      throw error.response?.data || { message: "Failed to fetch predictions" };
    }
  },

  // Get prediction by ID
  getPredictionById: async (predictionId) => {
    try {
      const response = await api.get(`/api/predictions/${predictionId}`);
      console.log("Prediction by ID response:", response);
      return response.data;
    } catch (error) {
      console.error(`Error fetching prediction ${predictionId}:`, error);
      throw (
        error.response?.data || {
          message: "Failed to fetch prediction details",
        }
      );
    }
  },

  // Get the axios instance
  getApi: () => {
    return api;
  },
};

export default apiService;
