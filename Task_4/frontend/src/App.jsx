import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm';
import ResultsDisplay from './components/ResultsDisplay';
import Dashboard from './components/Dashboard';
import apiService from './services/api';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('predict');

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await apiService.predict(formData);
      setPrediction(result);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while making the prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="header">
        <div className="container">
          <h1>🏪 Walmart Sales Forecasting</h1>
          <p>Advanced ML-powered sales prediction system</p>
        </div>
      </div>

      <div className="container">
        {/* Navigation Tabs */}
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'predict' ? 'active' : ''}`}
            onClick={() => setActiveTab('predict')}
          >
            Make Prediction
          </button>
          <button 
            className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            System Dashboard
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'predict' && (
          <>
            <PredictionForm onPredict={handlePredict} loading={loading} />
            <ResultsDisplay result={prediction} error={error} />
          </>
        )}

        {activeTab === 'dashboard' && (
          <Dashboard />
        )}

        {/* Footer */}
        <div className="footer">
          <div className="card">
            <h3>About This System</h3>
            <p>
              This professional sales forecasting system uses 6 advanced machine learning models 
              with 89 engineered features to predict weekly sales for Walmart stores. 
              The system achieves high accuracy with a Mean Absolute Error of $111.17.
            </p>
            <div className="stats">
              <div className="stat">
                <strong>Models:</strong> 6 Advanced ML Models
              </div>
              <div className="stat">
                <strong>Features:</strong> 89 Engineered Features
              </div>
              <div className="stat">
                <strong>Accuracy:</strong> $111.17 MAE
              </div>
              <div className="stat">
                <strong>Speed:</strong> 100+ predictions/sec
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .tabs {
          display: flex;
          margin-bottom: 20px;
          border-bottom: 2px solid #e9ecef;
        }

        .tab {
          background: none;
          border: none;
          padding: 12px 24px;
          cursor: pointer;
          font-size: 16px;
          color: #6c757d;
          border-bottom: 2px solid transparent;
          transition: all 0.3s ease;
        }

        .tab:hover {
          color: #007bff;
        }

        .tab.active {
          color: #007bff;
          border-bottom-color: #007bff;
          font-weight: 500;
        }

        .footer {
          margin-top: 40px;
        }

        .stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 15px;
          margin-top: 15px;
        }

        .stat {
          background: #f8f9fa;
          padding: 10px;
          border-radius: 4px;
          text-align: center;
        }

        .result-item {
          background: #f8f9fa;
          padding: 15px;
          border-radius: 6px;
          border-left: 4px solid #28a745;
        }

        .result-item h3 {
          margin-bottom: 8px;
          color: #495057;
          font-size: 14px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .prediction-summary {
          margin-top: 20px;
          padding: 15px;
          background: #e7f3ff;
          border-radius: 6px;
          border-left: 4px solid #007bff;
        }

        .prediction-summary h3 {
          margin-bottom: 10px;
          color: #007bff;
        }
      `}</style>
    </div>
  );
}

export default App;