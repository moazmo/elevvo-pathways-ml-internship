import React from 'react';
import { format } from 'date-fns';

const ResultsDisplay = ({ result, error }) => {
  if (error) {
    return (
      <div className="card">
        <div className="alert alert-error">
          <strong>Error:</strong> {error}
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatDate = (dateString) => {
    return format(new Date(dateString), 'MMMM dd, yyyy');
  };

  return (
    <div className="card">
      <h2>Prediction Results</h2>
      
      <div className="grid">
        <div className="result-item">
          <h3>Store & Department</h3>
          <p>Store #{result.store_id}, Department #{result.dept_id}</p>
        </div>

        <div className="result-item">
          <h3>Prediction Date</h3>
          <p>{formatDate(result.date)}</p>
        </div>

        <div className="result-item">
          <h3>Predicted Sales</h3>
          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#28a745' }}>
            {formatCurrency(result.predicted_sales)}
          </p>
        </div>

        <div className="result-item">
          <h3>Confidence Interval</h3>
          <p>
            {formatCurrency(result.confidence_interval[0])} - {formatCurrency(result.confidence_interval[1])}
          </p>
        </div>

        <div className="result-item">
          <h3>Model Used</h3>
          <p>{result.model_used.replace('_', ' ').toUpperCase()}</p>
        </div>

        <div className="result-item">
          <h3>Prediction Time</h3>
          <p>{format(new Date(result.prediction_timestamp), 'MMM dd, yyyy HH:mm:ss')}</p>
        </div>
      </div>

      <div className="prediction-summary">
        <h3>Summary</h3>
        <p>
          The model predicts weekly sales of <strong>{formatCurrency(result.predicted_sales)}</strong> for 
          Store #{result.store_id}, Department #{result.dept_id} on {formatDate(result.date)}.
          The confidence interval suggests sales could range from{' '}
          <strong>{formatCurrency(result.confidence_interval[0])}</strong> to{' '}
          <strong>{formatCurrency(result.confidence_interval[1])}</strong>.
        </p>
      </div>
    </div>
  );
};

export default ResultsDisplay;