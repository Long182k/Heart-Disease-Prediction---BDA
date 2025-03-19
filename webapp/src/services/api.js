import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Error handler for API requests
const handleApiError = (error) => {
  if (error.response) {
    // The request was made and the server responded with a status code outside the 2xx range
    console.error('API Error Response:', error.response.data);
    return Promise.reject(error.response.data);
  } else if (error.request) {
    // The request was made but no response was received
    console.error('API Error Request:', error.request);
    return Promise.reject({ message: 'No response from server. Please check your connection.' });
  } else {
    // Something happened in setting up the request
    console.error('API Error:', error.message);
    return Promise.reject({ message: error.message });
  }
};

// API service methods
const apiService = {
  // Get feature definitions for form
  getFeatureDefinitions: async () => {
    try {
      const response = await api.get('/features');
      return response.data.features;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Submit data for prediction
  makePrediction: async (data) => {
    try {
      const response = await api.post('/predict', data);
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  },

  // Get model metrics for display
  getModelMetrics: async () => {
    try {
      const response = await api.get('/metrics');
      return response.data;
    } catch (error) {
      return handleApiError(error);
    }
  }
};

export default apiService;
