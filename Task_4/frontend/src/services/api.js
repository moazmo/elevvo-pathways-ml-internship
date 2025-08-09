import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Health check
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // Get available models
  async getModels() {
    const response = await api.get('/models');
    return response.data;
  },

  // Single prediction
  async predict(predictionData) {
    const response = await api.post('/predict', predictionData);
    return response.data;
  },

  // Batch predictions
  async batchPredict(predictions) {
    const response = await api.post('/batch_predict', predictions);
    return response.data;
  },

  // Get system status
  async getStatus() {
    const response = await api.get('/');
    return response.data;
  }
};

export default apiService;