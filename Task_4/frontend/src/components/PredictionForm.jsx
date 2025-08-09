import React, { useState } from 'react';
import { format } from 'date-fns';

const PredictionForm = ({ onPredict, loading }) => {
  const [formData, setFormData] = useState({
    store_id: 1,
    dept_id: 1,
    date: format(new Date(), 'yyyy-MM-dd'),
    temperature: 70.0,
    fuel_price: 3.5,
    cpi: 220.0,
    unemployment: 7.0
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'store_id' || name === 'dept_id' ? parseInt(value) : 
              ['temperature', 'fuel_price', 'cpi', 'unemployment'].includes(name) ? 
              parseFloat(value) : value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredict(formData);
  };

  return (
    <div className="card">
      <h2>Sales Prediction</h2>
      <form onSubmit={handleSubmit}>
        <div className="grid">
          <div className="form-group">
            <label htmlFor="store_id">Store ID</label>
            <input
              type="number"
              id="store_id"
              name="store_id"
              value={formData.store_id}
              onChange={handleChange}
              min="1"
              max="45"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="dept_id">Department ID</label>
            <input
              type="number"
              id="dept_id"
              name="dept_id"
              value={formData.dept_id}
              onChange={handleChange}
              min="1"
              max="99"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="date">Date</label>
            <input
              type="date"
              id="date"
              name="date"
              value={formData.date}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="temperature">Temperature (°F)</label>
            <input
              type="number"
              id="temperature"
              name="temperature"
              value={formData.temperature}
              onChange={handleChange}
              step="0.1"
              min="-20"
              max="120"
            />
          </div>

          <div className="form-group">
            <label htmlFor="fuel_price">Fuel Price ($)</label>
            <input
              type="number"
              id="fuel_price"
              name="fuel_price"
              value={formData.fuel_price}
              onChange={handleChange}
              step="0.01"
              min="0"
              max="10"
            />
          </div>

          <div className="form-group">
            <label htmlFor="cpi">Consumer Price Index</label>
            <input
              type="number"
              id="cpi"
              name="cpi"
              value={formData.cpi}
              onChange={handleChange}
              step="0.1"
              min="100"
              max="300"
            />
          </div>

          <div className="form-group">
            <label htmlFor="unemployment">Unemployment Rate (%)</label>
            <input
              type="number"
              id="unemployment"
              name="unemployment"
              value={formData.unemployment}
              onChange={handleChange}
              step="0.1"
              min="0"
              max="20"
            />
          </div>
        </div>

        <button type="submit" className="btn" disabled={loading}>
          {loading ? 'Predicting...' : 'Get Prediction'}
        </button>
      </form>
    </div>
  );
};

export default PredictionForm;