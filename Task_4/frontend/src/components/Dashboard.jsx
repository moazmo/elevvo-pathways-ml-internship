import React, { useState, useEffect } from 'react';
import apiService from '../services/api';

const Dashboard = () => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [healthData, setHealthData] = useState(null);
  const [modelsData, setModelsData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const [status, health, models] = await Promise.all([
          apiService.getStatus(),
          apiService.getHealth(),
          apiService.getModels()
        ]);
        
        setSystemStatus(status);
        setHealthData(health);
        setModelsData(models);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="loading">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>System Dashboard</h2>
      
      <div className="grid">
        {/* System Status */}
        <div className="dashboard-item">
          <h3>System Status</h3>
          <div className="status-indicator">
            <span className={`status-dot ${systemStatus?.status === 'running' ? 'green' : 'red'}`}></span>
            <span>{systemStatus?.status || 'Unknown'}</span>
          </div>
          <p>Models Loaded: {systemStatus?.models_loaded || 0}</p>
          <p>Features Available: {systemStatus?.features_available || 0}</p>
        </div>

        {/* Health Check */}
        <div className="dashboard-item">
          <h3>Health Check</h3>
          <div className="status-indicator">
            <span className={`status-dot ${healthData?.status === 'healthy' ? 'green' : 'red'}`}></span>
            <span>{healthData?.status || 'Unknown'}</span>
          </div>
          <p>Last Check: {healthData?.timestamp ? new Date(healthData.timestamp).toLocaleString() : 'N/A'}</p>
          <p>Fallback Mode: {healthData?.models?.fallback_mode ? 'Yes' : 'No'}</p>
        </div>

        {/* Models Information */}
        <div className="dashboard-item">
          <h3>Available Models</h3>
          <p>Total Models: {modelsData?.total_models || 0}</p>
          <p>Feature Count: {modelsData?.feature_count || 0}</p>
          {modelsData?.available_models && modelsData.available_models.length > 0 && (
            <div>
              <strong>Models:</strong>
              <ul>
                {modelsData.available_models.map((model, index) => (
                  <li key={index}>{model.replace('_', ' ').toUpperCase()}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Performance Metrics */}
        <div className="dashboard-item">
          <h3>Performance</h3>
          <p>Best Model: Weighted Ensemble</p>
          <p>Accuracy: $111.17 MAE</p>
          <p>Speed: 100+ predictions/sec</p>
          <p>Uptime: {healthData?.timestamp ? 'Active' : 'Unknown'}</p>
        </div>
      </div>

      <style jsx>{`
        .dashboard-item {
          background: #f8f9fa;
          padding: 15px;
          border-radius: 6px;
          border-left: 4px solid #007bff;
        }

        .status-indicator {
          display: flex;
          align-items: center;
          margin-bottom: 10px;
        }

        .status-dot {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          margin-right: 8px;
        }

        .status-dot.green {
          background-color: #28a745;
        }

        .status-dot.red {
          background-color: #dc3545;
        }

        .dashboard-item ul {
          margin-top: 5px;
          padding-left: 20px;
        }

        .dashboard-item li {
          margin-bottom: 2px;
        }
      `}</style>
    </div>
  );
};

export default Dashboard;