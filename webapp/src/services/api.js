import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const ApiService = {
  predict: async (patientData) => {
    try {
      const response = await axios.post(`${API_URL}/predict`, patientData);
      return response.data;
    } catch (error) {
      console.error('Error predicting heart disease:', error);
      throw error;
    }
  },
  
  getMetrics: async () => {
    try {
      const response = await axios.get(`${API_URL}/metrics`);
      return response.data;
    } catch (error) {
      console.error('Error fetching metrics:', error);
      throw error;
    }
  },
  
  getFeatures: async () => {
    try {
      const response = await axios.get(`${API_URL}/features`);
      return response.data.features;
    } catch (error) {
      console.error('Error fetching features:', error);
      throw error;
    }
  }
};

export default ApiService;
