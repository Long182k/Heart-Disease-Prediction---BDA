import axios from "axios";

const AUTH_TOKEN_KEY = "cardio_auth_token";
const USER_KEY = "cardio_user";

// Create a base axios instance for auth requests
const authAxios = axios.create();

const authService = {
  // Register a new user
  register: async (userData) => {
    try {
      const response = await authAxios.post("/api/auth/register", userData);
      console.log("🚀 response:", response);

      // The API returns the actual data inside response.data
      const responseData = response.data;
      
      // Check if the response indicates success (status 201)
      if (response.status === 201 && responseData.token) {
        // Store token
        localStorage.setItem(AUTH_TOKEN_KEY, responseData.token);
        localStorage.setItem(USER_KEY, JSON.stringify(responseData.user));
        
        // Set the token in the authAxios headers for future requests
        authAxios.defaults.headers.common["Authorization"] = `Bearer ${responseData.token}`;
        
        return responseData;
      } else {
        // If the API returns a response without a token
        throw new Error(responseData.message || "Registration failed");
      }
    } catch (error) {
      console.error(
        "Registration error:",
        error.response?.data || error.message
      );
      throw error.response?.data || { message: "Registration failed" };
    }
  },

  // Login a user
  login: async (credentials) => {
    try {
      const response = await authAxios.post("/api/auth/login", credentials);
      if (response.data.token) {
        localStorage.setItem(AUTH_TOKEN_KEY, response.data.token);
        localStorage.setItem(USER_KEY, JSON.stringify(response.data.user));

        // Set the token in the authAxios headers for future requests
        authAxios.defaults.headers.common["Authorization"] = `Bearer ${response.data.token}`;
      }
      return response.data;
    } catch (error) {
      console.error("Login error:", error.response?.data || error.message);
      throw error.response?.data || { message: "Login failed" };
    }
  },

  // Logout the current user
  logout: () => {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    delete authAxios.defaults.headers.common["Authorization"];
  },

  // Check if user is authenticated
  isAuthenticated: () => {
    return !!localStorage.getItem(AUTH_TOKEN_KEY);
  },

  // Get the current user
  getCurrentUser: () => {
    const userStr = localStorage.getItem(USER_KEY);
    return userStr ? JSON.parse(userStr) : null;
  },

  // Get the auth token
  getToken: () => {
    return localStorage.getItem(AUTH_TOKEN_KEY);
  },

  // Initialize auth state from localStorage
  initializeAuth: () => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    if (token) {
      authAxios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
    }
  },
  
  // Get the axios instance with auth headers
  getAuthAxios: () => {
    return authAxios;
  }
};

// Initialize auth on service import
authService.initializeAuth();

export default authService;
