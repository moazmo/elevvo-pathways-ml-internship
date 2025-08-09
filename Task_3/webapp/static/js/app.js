// Loan Approval System - Frontend JavaScript

class LoanApprovalApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8001';
        this.predictionsToday = 0;
        this.recentPredictions = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupNavigation();
        this.loadDashboardData();
        this.setupFormValidation();
    }

    setupEventListeners() {
        console.log('🔧 Setting up event listeners...');

        // Form submission
        const form = document.getElementById('loanForm');
        const predictBtn = document.getElementById('predictBtn');

        if (form) {
            form.addEventListener('submit', (e) => {
                console.log('📝 Form submit event triggered');
                e.preventDefault();
                this.handlePrediction();
            });
            console.log('✅ Form submit listener added');
        } else {
            console.error('❌ Form not found!');
        }

        if (predictBtn) {
            predictBtn.addEventListener('click', (e) => {
                console.log('🔘 Predict button clicked');
                e.preventDefault();
                this.handlePrediction();
            });
            console.log('✅ Predict button listener added');
        } else {
            console.error('❌ Predict button not found!');
        }

        // Clear form
        const clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                console.log('🧹 Clear button clicked');
                this.clearForm();
            });
        }

        // Batch processing
        const batchBtn = document.getElementById('processBatchBtn');
        if (batchBtn) {
            batchBtn.addEventListener('click', () => {
                this.handleBatchProcessing();
            });
        }

        // Download template
        const templateLink = document.getElementById('downloadTemplate');
        if (templateLink) {
            templateLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.downloadTemplate();
            });
        }

        // Download results
        const downloadBtn = document.getElementById('downloadResults');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadBatchResults();
            });
        }

        console.log('✅ All event listeners set up');
    }

    setupNavigation() {
        // Navigation handling
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = e.target.getAttribute('href');
                
                if (target === '#api-docs') {
                    window.open(`${this.apiBaseUrl}/docs`, '_blank');
                    return;
                }

                this.showSection(target.substring(1));
                
                // Update active nav
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                e.target.classList.add('active');
            });
        });
    }

    showSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.style.display = 'none';
        });
        
        // Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.style.display = 'block';
            targetSection.classList.add('active');
        }

        // Load section-specific data
        if (sectionId === 'dashboard') {
            this.loadDashboardData();
        }
    }

    setupFormValidation() {
        // Real-time validation
        const form = document.getElementById('loanForm');
        const inputs = form.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateField(input);
            });
        });
    }

    validateField(field) {
        const value = field.value.trim();
        const fieldName = field.name;

        // Remove existing validation classes
        field.classList.remove('is-valid', 'is-invalid');

        // Validation rules
        let isValid = true;
        let errorMessage = '';

        if (!value) {
            isValid = false;
            errorMessage = 'This field is required';
        } else {
            switch (fieldName) {
                case 'cibil_score':
                    const score = parseInt(value);
                    if (score < 300 || score > 900) {
                        isValid = false;
                        errorMessage = 'CIBIL score must be between 300-900';
                    }
                    break;
                case 'income_annum':
                case 'loan_amount':
                case 'residential_assets_value':
                case 'commercial_assets_value':
                case 'luxury_assets_value':
                case 'bank_asset_value':
                    if (parseFloat(value) < 0) {
                        isValid = false;
                        errorMessage = 'Amount cannot be negative';
                    }
                    break;
                case 'no_of_dependents':
                case 'loan_term':
                    if (parseInt(value) < 0) {
                        isValid = false;
                        errorMessage = 'Value cannot be negative';
                    }
                    break;
            }
        }

        // Apply validation styling
        field.classList.add(isValid ? 'is-valid' : 'is-invalid');

        // Show/hide error message
        let feedback = field.parentNode.querySelector('.invalid-feedback');
        if (!isValid) {
            if (!feedback) {
                feedback = document.createElement('div');
                feedback.className = 'invalid-feedback';
                field.parentNode.appendChild(feedback);
            }
            feedback.textContent = errorMessage;
        } else if (feedback) {
            feedback.remove();
        }

        console.log(`Field ${fieldName}: ${isValid ? 'valid' : 'invalid'} (value: ${value})`);
        return isValid;
    }

    async handlePrediction() {
        console.log('🚀 Starting prediction process...');

        const form = document.getElementById('loanForm');
        if (!form) {
            console.error('❌ Form not found!');
            this.showAlert('Form not found. Please refresh the page.', 'danger');
            return;
        }

        // Show loading immediately
        this.showLoading(true);

        try {
            const formData = new FormData(form);
            console.log('📝 Form data collected');

            // Validate all fields
            const inputs = form.querySelectorAll('input[required], select[required]');
            let allValid = true;
            let missingFields = [];

            inputs.forEach(input => {
                if (!input.value || input.value.trim() === '') {
                    allValid = false;
                    missingFields.push(input.name || input.id);
                    console.log(`❌ Missing field: ${input.name || input.id}`);
                }
            });

            if (!allValid) {
                console.log('❌ Validation failed. Missing fields:', missingFields);
                this.showAlert(`Please fill in all required fields: ${missingFields.join(', ')}`, 'warning');
                return;
            }

            // Prepare data with explicit type conversion
            const applicationData = {};

            // Text fields
            applicationData.loan_id = formData.get('loan_id') || `LOAN_${Date.now()}`;
            applicationData.education = formData.get('education');
            applicationData.self_employed = formData.get('self_employed');

            // Numeric fields with validation
            const numericFields = {
                'no_of_dependents': 'Number of Dependents',
                'income_annum': 'Annual Income',
                'loan_amount': 'Loan Amount',
                'loan_term': 'Loan Term',
                'cibil_score': 'CIBIL Score',
                'residential_assets_value': 'Residential Assets',
                'commercial_assets_value': 'Commercial Assets',
                'luxury_assets_value': 'Luxury Assets',
                'bank_asset_value': 'Bank Assets'
            };

            for (const [field, label] of Object.entries(numericFields)) {
                const value = formData.get(field);
                const numValue = parseFloat(value);

                if (isNaN(numValue) || numValue < 0) {
                    console.log(`❌ Invalid ${field}: ${value}`);
                    this.showAlert(`Invalid value for ${label}. Please enter a valid positive number.`, 'danger');
                    return;
                }

                applicationData[field] = numValue;
            }

            console.log('✅ Data prepared:', applicationData);

            // Make API request
            console.log('🌐 Sending request to API...');
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(applicationData)
            });

            console.log('📡 Response received. Status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('❌ API Error:', errorText);
                throw new Error(`Server error (${response.status}): ${errorText}`);
            }

            const result = await response.json();
            console.log('✅ Prediction successful:', result);

            this.displayResults(result);
            this.updatePredictionStats(result);

        } catch (error) {
            console.error('💥 Prediction error:', error);
            this.showAlert(`Prediction failed: ${error.message}`, 'danger');
        } finally {
            // Always hide loading
            console.log('🔄 Hiding loading modal...');
            this.showLoading(false);
        }
    }

    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsCard = document.getElementById('resultsCard');
        const resultsContent = document.getElementById('resultsContent');

        // Set card color based on prediction
        const isApproved = result.prediction === 'Approved';
        resultsCard.className = `card shadow-lg ${isApproved ? 'result-approved' : 'result-rejected'}`;

        // Create results HTML
        const confidencePercent = (result.confidence * 100).toFixed(1);
        const riskPercent = (result.risk_score * 100).toFixed(1);

        resultsContent.innerHTML = `
            <div class="row text-center">
                <div class="col-md-4">
                    <div class="mb-3">
                        <i class="fas ${isApproved ? 'fa-check-circle' : 'fa-times-circle'} fa-4x mb-3"></i>
                        <h3>${result.prediction}</h3>
                        <p class="mb-0">Loan Application</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="mb-3">
                        <h4>${confidencePercent}%</h4>
                        <div class="confidence-bar mb-2">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                        <p class="mb-0">Confidence Level</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="mb-3">
                        <div class="risk-gauge">
                            <div class="risk-gauge-inner">
                                ${riskPercent}%
                            </div>
                        </div>
                        <p class="mb-0 mt-2">Risk Score</p>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-12">
                    <div class="alert ${isApproved ? 'alert-success' : 'alert-danger'} text-center">
                        <h5 class="mb-2">
                            <i class="fas ${isApproved ? 'fa-thumbs-up' : 'fa-thumbs-down'} me-2"></i>
                            ${isApproved ? 'Congratulations!' : 'Application Not Approved'}
                        </h5>
                        <p class="mb-0">
                            ${isApproved 
                                ? 'Your loan application meets our approval criteria.' 
                                : 'Your application does not meet the current approval criteria.'}
                        </p>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-12 text-center">
                    <button type="button" class="btn btn-outline-light btn-lg" onclick="app.clearResults()">
                        <i class="fas fa-redo me-2"></i>
                        New Application
                    </button>
                </div>
            </div>
        `;

        // Show results with animation
        resultsSection.style.display = 'block';
        resultsSection.classList.add('result-animation');
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    clearResults() {
        document.getElementById('resultsSection').style.display = 'none';
        this.clearForm();
    }

    clearForm() {
        document.getElementById('loanForm').reset();
        
        // Remove validation classes
        document.querySelectorAll('.form-control, .form-select').forEach(field => {
            field.classList.remove('is-valid', 'is-invalid');
        });
        
        // Remove error messages
        document.querySelectorAll('.invalid-feedback').forEach(feedback => {
            feedback.remove();
        });
    }

    async loadDashboardData() {
        try {
            // Load system health
            const healthResponse = await fetch(`${this.apiBaseUrl}/health`);
            const healthData = await healthResponse.json();
            document.getElementById('systemStatus').textContent = healthData.status;

            // Load model info
            const modelResponse = await fetch(`${this.apiBaseUrl}/model/info`);
            const modelData = await modelResponse.json();
            document.getElementById('activeModel').textContent = modelData.model_type;
            document.getElementById('modelAccuracy').textContent = '100%';

            // Update predictions count
            document.getElementById('predictionsToday').textContent = this.predictionsToday;

            // Load performance chart
            this.loadPerformanceChart();

        } catch (error) {
            console.error('Dashboard loading error:', error);
        }
    }

    loadPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;

        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                datasets: [{
                    label: 'Decision Tree',
                    data: [100, 100, 100, 100, 100],
                    borderColor: 'rgb(13, 110, 253)',
                    backgroundColor: 'rgba(13, 110, 253, 0.2)',
                    pointBackgroundColor: 'rgb(13, 110, 253)',
                }, {
                    label: 'Random Forest',
                    data: [99.88, 99.69, 100, 99.85, 99.99],
                    borderColor: 'rgb(25, 135, 84)',
                    backgroundColor: 'rgba(25, 135, 84, 0.2)',
                    pointBackgroundColor: 'rgb(25, 135, 84)',
                }]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    updatePredictionStats(result) {
        this.predictionsToday++;
        this.recentPredictions.unshift({
            loan_id: result.loan_id,
            prediction: result.prediction,
            confidence: result.confidence,
            timestamp: new Date().toLocaleTimeString()
        });

        // Keep only last 10 predictions
        if (this.recentPredictions.length > 10) {
            this.recentPredictions = this.recentPredictions.slice(0, 10);
        }

        // Update dashboard if visible
        if (document.getElementById('dashboard').style.display !== 'none') {
            this.updateRecentPredictions();
        }
    }

    updateRecentPredictions() {
        const container = document.getElementById('recentPredictions');
        
        if (this.recentPredictions.length === 0) {
            container.innerHTML = '<p class="text-muted">No recent predictions</p>';
            return;
        }

        const html = this.recentPredictions.map(pred => `
            <div class="d-flex justify-content-between align-items-center mb-2 p-2 border rounded">
                <div>
                    <strong>${pred.loan_id}</strong><br>
                    <small class="text-muted">${pred.timestamp}</small>
                </div>
                <div class="text-end">
                    <span class="badge ${pred.prediction === 'Approved' ? 'bg-success' : 'bg-danger'}">
                        ${pred.prediction}
                    </span><br>
                    <small>${(pred.confidence * 100).toFixed(1)}%</small>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    async handleBatchProcessing() {
        const fileInput = document.getElementById('csvFile');
        const file = fileInput.files[0];

        if (!file) {
            this.showAlert('Please select a CSV file', 'warning');
            return;
        }

        if (!file.name.endsWith('.csv')) {
            this.showAlert('Please select a valid CSV file', 'danger');
            return;
        }

        this.showLoading(true);

        try {
            // Parse CSV file
            const csvText = await file.text();
            const applications = this.parseCSV(csvText);

            // Send batch request
            const response = await fetch(`${this.apiBaseUrl}/predict/batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(applications)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            this.displayBatchResults(results);

        } catch (error) {
            console.error('Batch processing error:', error);
            this.showAlert('Error processing batch file. Please check the format.', 'danger');
        } finally {
            this.showLoading(false);
        }
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        const applications = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            const app = {};
            
            headers.forEach((header, index) => {
                const value = values[index]?.trim();
                
                // Convert numeric fields
                if (['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                     'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
                     'bank_asset_value'].includes(header)) {
                    app[header] = parseFloat(value) || 0;
                } else {
                    app[header] = value;
                }
            });
            
            applications.push(app);
        }
        
        return applications;
    }

    displayBatchResults(results) {
        const batchResults = document.getElementById('batchResults');
        const tableBody = document.querySelector('#batchResultsTable tbody');
        
        // Clear previous results
        tableBody.innerHTML = '';
        
        // Populate table
        results.forEach(result => {
            const row = tableBody.insertRow();
            row.innerHTML = `
                <td>${result.loan_id}</td>
                <td>
                    <span class="badge ${result.prediction === 'Approved' ? 'bg-success' : 'bg-danger'}">
                        ${result.prediction}
                    </span>
                </td>
                <td>${(result.confidence * 100).toFixed(1)}%</td>
                <td>${(result.risk_score * 100).toFixed(1)}%</td>
            `;
        });

        // Show results
        batchResults.style.display = 'block';
        batchResults.scrollIntoView({ behavior: 'smooth' });

        // Store results for download
        this.batchResultsData = results;
    }

    downloadTemplate() {
        const template = `loan_id,no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value
SAMPLE_001,2,Graduate,No,6000000,15000000,15,750,2500000,1200000,600000,400000
SAMPLE_002,1,Graduate,No,8000000,12000000,10,820,3000000,1500000,800000,500000`;

        this.downloadFile(template, 'loan_application_template.csv', 'text/csv');
    }

    downloadBatchResults() {
        if (!this.batchResultsData) {
            this.showAlert('No batch results to download', 'warning');
            return;
        }

        // Convert to CSV
        const headers = ['loan_id', 'prediction', 'confidence', 'risk_score', 'timestamp'];
        const csvContent = [
            headers.join(','),
            ...this.batchResultsData.map(result => [
                result.loan_id,
                result.prediction,
                result.confidence.toFixed(4),
                result.risk_score.toFixed(4),
                new Date().toISOString()
            ].join(','))
        ].join('\n');

        this.downloadFile(csvContent, 'batch_predictions_results.csv', 'text/csv');
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    showLoading(show) {
        console.log(`🔄 Loading modal: ${show ? 'SHOW' : 'HIDE'}`);

        const modalElement = document.getElementById('loadingModal');
        if (!modalElement) {
            console.error('❌ Loading modal element not found!');
            return;
        }

        if (show) {
            try {
                // Show loading modal
                console.log('📱 Showing loading modal...');
                const modal = new bootstrap.Modal(modalElement, {
                    backdrop: 'static',
                    keyboard: false
                });
                modal.show();
                this.currentModal = modal;
                console.log('✅ Loading modal shown');
            } catch (error) {
                console.error('❌ Error showing modal:', error);
            }
        } else {
            try {
                console.log('📱 Hiding loading modal...');

                // Method 1: Use stored modal instance
                if (this.currentModal) {
                    this.currentModal.hide();
                    this.currentModal = null;
                    console.log('✅ Modal hidden via instance');
                }

                // Method 2: Force hide using Bootstrap API
                const modalInstance = bootstrap.Modal.getInstance(modalElement);
                if (modalInstance) {
                    modalInstance.hide();
                    console.log('✅ Modal hidden via Bootstrap instance');
                }

                // Method 3: Force hide manually (fallback)
                setTimeout(() => {
                    modalElement.classList.remove('show');
                    modalElement.style.display = 'none';
                    modalElement.setAttribute('aria-hidden', 'true');
                    modalElement.removeAttribute('aria-modal');

                    // Remove all backdrops
                    document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
                        backdrop.remove();
                    });

                    // Restore body
                    document.body.classList.remove('modal-open');
                    document.body.style.overflow = '';
                    document.body.style.paddingRight = '';

                    console.log('✅ Modal force hidden');
                }, 100);

            } catch (error) {
                console.error('❌ Error hiding modal:', error);
            }
        }
    }

    showAlert(message, type = 'info') {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new LoanApprovalApp();
});

// Utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 0
    }).format(amount);
}

function formatNumber(number) {
    return new Intl.NumberFormat('en-IN').format(number);
}
