// Loan Approval AI - Optimized for Speed & Functionality
// =====================================================

// SCHEMA CONFIGURATION
const SCHEMA = {
    target: 'loan_approved',
    features: ['income', 'credit_score', 'loan_amount', 'years_employed', 'points'],
    identifier: 'name',
    quantitative: ['income', 'credit_score', 'loan_amount', 'years_employed', 'points'],
    qualitative: [],
    derivedFeatures: {
        'debt_to_income': (row) => row.loan_amount / (row.income || 1)
    }
};

// Global application state
const appState = {
    rawData: [],
    testData: [],
    processedData: null,
    model: null,
    trainingHistory: null,
    validationPredictions: null,
    validationLabels: null,
    featureImportance: null,
    charts: {},
    currentChart: null,
    testPredictions: null,
    preprocessingStats: null,
    currentMetrics: null,
    trainIndices: [],
    valIndices: []
};

// DOM elements
const elements = {
    dataStatus: () => document.getElementById('dataStatus'),
    preprocessStatus: () => document.getElementById('preprocessStatus'),
    modelStatus: () => document.getElementById('modelStatus'),
    trainingStatus: () => document.getElementById('trainingStatus'),
    predictionStatus: () => document.getElementById('predictionStatus'),
    dataPreview: () => document.getElementById('dataPreview'),
    featureInfo: () => document.getElementById('featureInfo'),
    modelSummary: () => document.getElementById('modelSummary'),
    trainingProgress: () => document.getElementById('trainingProgress'),
    thresholdValue: () => document.getElementById('thresholdValue'),
    featureImportance: () => document.getElementById('featureImportance'),
    edaInsights: () => document.getElementById('edaInsights'),
    chartsContainer: () => document.getElementById('chartsContainer'),
    chartControls: () => document.getElementById('chartControls'),
    chartDisplay: () => document.getElementById('chartDisplay'),
    predictionResults: () => document.getElementById('predictionResults'),
    preprocessingDetails: () => document.getElementById('preprocessingDetails')
};

// ==================== UTILITY FUNCTIONS ====================

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('File reading failed'));
        reader.readAsText(file);
    });
}

function parseCSV(csvText) {
    const firstLine = csvText.split('\n')[0];
    let delimiter = ';';
    
    if (firstLine.includes(',')) {
        delimiter = ',';
    }
    
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(delimiter).map(h => h.trim());
    
    return lines.slice(1).filter(line => line.trim() !== '').map(line => {
        const values = line.split(delimiter).map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
            let value = values[index];
            
            if (index === 0 && header.charCodeAt(0) === 65279) {
                header = header.substring(1);
            }
            
            if (!isNaN(value) && value !== '') {
                value = Number(value);
            }
            else if (value === 'true' || value === '1' || value === 'True') {
                value = true;
            }
            else if (value === 'false' || value === '0' || value === 'False') {
                value = false;
            }
            
            row[header] = value;
        });
        return row;
    });
}

function validateSchema(data) {
    if (data.length === 0) {
        throw new Error('No data loaded');
    }
    
    const firstRow = data[0];
    
    SCHEMA.features.forEach(feature => {
        if (!(feature in firstRow)) {
            throw new Error(`Missing required feature: ${feature}. Available: ${Object.keys(firstRow).join(', ')}`);
        }
    });
    
    if (!(SCHEMA.target in firstRow)) {
        throw new Error(`Missing target variable: ${SCHEMA.target}. Available: ${Object.keys(firstRow).join(', ')}`);
    }
}

function showStatus(element, message, type = '') {
    element.textContent = message;
    element.className = `status ${type}`;
}

function updateUIState() {
    const buttons = ['preprocessBtn', 'createModelBtn', 'trainBtn', 'predictBtn', 'exportBtn', 'downloadPredictionsBtn', 'loadModelBtn'];
    buttons.forEach(btn => {
        const element = document.getElementById(btn);
        if (element) {
            if (btn === 'preprocessBtn') element.disabled = appState.rawData.length === 0;
            if (btn === 'createModelBtn') element.disabled = !appState.processedData;
            if (btn === 'trainBtn') element.disabled = !appState.model;
            if (btn === 'predictBtn') element.disabled = !appState.model || appState.testData.length === 0;
            if (btn === 'exportBtn') element.disabled = !appState.model;
            if (btn === 'downloadPredictionsBtn') element.disabled = !appState.testPredictions;
            if (btn === 'loadModelBtn') element.disabled = false;
        }
    });
}

function displayDataPreview() {
    const container = elements.dataPreview();
    const data = appState.rawData.slice(0, 10);
    
    if (data.length === 0) {
        container.innerHTML = '<p>No data to display</p>';
        return;
    }
    
    const headers = Object.keys(data[0]);
    let html = `<table>
        <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
        <tbody>`;
    
    data.forEach(row => {
        html += `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ==================== EDA & VISUALIZATION FUNCTIONS ====================

function performComprehensiveEDA() {
    const stats = calculateEnhancedStatistics(appState.rawData);
    displayDataPreview();
    createInteractiveCharts(appState.rawData);
    
    const insights = generateDetailedEDAInsights(stats);
    elements.edaInsights().innerHTML = insights;
}

function calculateEnhancedStatistics(data) {
    const targetValues = data.map(row => row[SCHEMA.target]);
    const approved = targetValues.filter(val => val === true || val === 1).length;
    const rejected = targetValues.filter(val => val === false || val === 0).length;
    
    const missingValues = {};
    SCHEMA.features.forEach(feature => {
        missingValues[feature] = data.filter(row => 
            row[feature] === null || row[feature] === undefined || row[feature] === ''
        ).length;
    });
    
    const numericalStats = {};
    SCHEMA.quantitative.forEach(feature => {
        const values = data.map(row => row[feature]).filter(val => val != null && val !== '');
        if (values.length > 0) {
            numericalStats[feature] = {
                min: Math.min(...values),
                max: Math.max(...values),
                mean: values.reduce((a, b) => a + b, 0) / values.length,
                median: calculateMedian(values),
                missing: missingValues[feature],
                missingPercentage: ((missingValues[feature] / data.length) * 100).toFixed(1)
            };
        }
    });
    
    return {
        totalSamples: data.length,
        approvalRate: (approved / data.length * 100).toFixed(1),
        approvedCount: approved,
        rejectedCount: rejected,
        numericalStats: numericalStats,
        missingValues: missingValues
    };
}

function calculateMedian(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function createInteractiveCharts(data) {
    // Clear previous charts
    elements.chartControls().innerHTML = '';
    elements.chartDisplay().innerHTML = '';
    
    // Properly destroy existing charts
    Object.values(appState.charts).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
    appState.charts = {};
    
    // Create chart buttons
    const chartTypes = ['Target Distribution', ...SCHEMA.quantitative];
    
    chartTypes.forEach((type, index) => {
        const button = document.createElement('button');
        button.className = `chart-btn ${index === 0 ? 'active' : ''}`;
        button.textContent = type;
        button.onclick = () => showChart(type);
        elements.chartControls().appendChild(button);
    });
    
    // Create all charts
    createTargetDistributionChart(data);
    SCHEMA.quantitative.forEach(feature => {
        createFeatureDistributionChart(data, feature);
    });
    
    // Show first chart
    showChart('Target Distribution');
}

function showChart(chartType) {
    // Update button states
    document.querySelectorAll('.chart-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent === chartType) {
            btn.classList.add('active');
        }
    });
    
    // Clear display and create new canvas
    elements.chartDisplay().innerHTML = '';
    const canvas = document.createElement('canvas');
    canvas.width = 500;
    canvas.height = 300;
    elements.chartDisplay().appendChild(canvas);
    
    // Create the chart on the canvas
    if (appState.charts[chartType] && appState.charts[chartType].config) {
        const ctx = canvas.getContext('2d');
        appState.charts[chartType].instance = new Chart(ctx, appState.charts[chartType].config);
    }
}

function createTargetDistributionChart(data) {
    const targetValues = data.map(row => row[SCHEMA.target]);
    const approved = targetValues.filter(val => val === true || val === 1).length;
    const rejected = targetValues.filter(val => val === false || val === 0).length;
    
    const config = {
        type: 'doughnut',
        data: {
            labels: ['Approved', 'Rejected'],
            datasets: [{
                data: [approved, rejected],
                backgroundColor: ['#27ae60', '#e74c3c'],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Loan Approval Distribution'
                },
                legend: {
                    position: 'bottom'
                }
            }
        }
    };
    
    appState.charts['Target Distribution'] = { config: config };
}

function createFeatureDistributionChart(data, feature) {
    const values = data.map(row => row[feature]).filter(v => v != null);
    
    // Create bins for histogram
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binCount = Math.min(15, Math.floor(values.length / 20));
    const binSize = (max - min) / binCount;
    
    const bins = Array(binCount).fill(0);
    values.forEach(value => {
        const binIndex = Math.min(binCount - 1, Math.floor((value - min) / binSize));
        bins[binIndex]++;
    });
    
    const config = {
        type: 'bar',
        data: {
            labels: bins.map((_, i) => {
                const start = min + i * binSize;
                const end = min + (i + 1) * binSize;
                return `${Math.round(start)}-${Math.round(end)}`;
            }),
            datasets: [{
                label: `Count`,
                data: bins,
                backgroundColor: '#3498db',
                borderColor: '#2980b9',
                borderWidth: 1
            }]
        },
        options: {
            responsive: false,
            plugins: {
                title: {
                    display: true,
                    text: `${feature} Distribution`
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: feature
                    }
                }
            }
        }
    };
    
    appState.charts[feature] = { config: config };
}

function generateDetailedEDAInsights(stats) {
    let insights = `<strong>🔍 Detailed EDA Analysis:</strong><div style="margin-top: 10px;">`;
    
    insights += `<div style="margin-bottom: 10px;">
        <strong>Dataset Overview:</strong><br>
        • Total samples: ${stats.totalSamples}<br>
        • Approval rate: ${stats.approvalRate}% (${stats.approvedCount} approved, ${stats.rejectedCount} rejected)
    </div>`;
    
    let hasMissing = false;
    insights += `<div style="margin-bottom: 10px;">
        <strong>Missing Values Analysis:</strong><br>`;
    
    SCHEMA.features.forEach(feature => {
        if (stats.missingValues[feature] > 0) {
            insights += `• ${feature}: ${stats.missingValues[feature]} (${stats.numericalStats[feature]?.missingPercentage || '0'}%)<br>`;
            hasMissing = true;
        }
    });
    
    if (!hasMissing) {
        insights += `• No missing values detected<br>`;
    }
    insights += `</div>`;
    
    insights += `<div>
        <strong>Business Insights:</strong><br>`;
    
    if (stats.approvalRate < 30) insights += "• Low approval rate suggests strict lending criteria<br>";
    if (stats.approvalRate > 70) insights += "• High approval rate indicates lenient policies<br>";
    if (stats.numericalStats.credit_score?.mean > 700) insights += "• Applicants generally have good credit scores<br>";
    
    if (!insights.includes("Business Insights")) {
        insights += "• Balanced dataset with diverse applicant profiles";
    }
    
    insights += `</div></div>`;
    return insights;
}

// ==================== DATA SPLITTING & PREPROCESSING ====================

function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

function splitDataIndices(data, validationSplit = 0.2) {
    const indices = Array.from({ length: data.length }, (_, i) => i);
    const shuffledIndices = shuffleArray(indices);
    
    const valCount = Math.floor(data.length * validationSplit);
    const valIndices = shuffledIndices.slice(0, valCount);
    const trainIndices = shuffledIndices.slice(valCount);
    
    return { trainIndices, valIndices };
}

function preprocessData() {
    try {
        showStatus(elements.preprocessStatus(), 'Engineering features and preprocessing...', 'loading');
        
        // Split data before any preprocessing
        const { trainIndices, valIndices } = splitDataIndices(appState.rawData, 0.2);
        appState.trainIndices = trainIndices;
        appState.valIndices = valIndices;
        
        // Extract features for training and validation
        const trainFeatures = trainIndices.map(index => {
            const row = appState.rawData[index];
            const featureRow = {};
            SCHEMA.features.forEach(feature => {
                featureRow[feature] = row[feature];
            });
            return featureRow;
        });
        
        const valFeatures = valIndices.map(index => {
            const row = appState.rawData[index];
            const featureRow = {};
            SCHEMA.features.forEach(feature => {
                featureRow[feature] = row[feature];
            });
            return featureRow;
        });
        
        const trainTargets = trainIndices.map(index => appState.rawData[index][SCHEMA.target] ? 1 : 0);
        const valTargets = valIndices.map(index => appState.rawData[index][SCHEMA.target] ? 1 : 0);
        
        // Track preprocessing statistics
        appState.preprocessingStats = {
            originalFeatures: SCHEMA.features.length,
            missingValues: {},
            standardization: {}
        };
        
        // Process training data and calculate statistics
        const processedTrainFeatures = imputeMissingValues(trainFeatures);
        const engineeredTrainFeatures = engineerNewFeatures(processedTrainFeatures);
        
        const quantitativeTrainData = engineeredTrainFeatures.map(row => 
            [...SCHEMA.quantitative, ...Object.keys(SCHEMA.derivedFeatures)].map(feature => row[feature])
        );
        
        const { standardizedData: standardizedTrainData, means, stds } = standardizeDataWithStats(quantitativeTrainData);
        
        // Store standardization stats from training data only
        SCHEMA.quantitative.forEach((feature, index) => {
            appState.preprocessingStats.standardization[feature] = {
                mean: means[index],
                std: stds[index]
            };
        });
        
        // Process validation data using training statistics
        const processedValFeatures = imputeMissingValues(valFeatures, true);
        const engineeredValFeatures = engineerNewFeatures(processedValFeatures);
        
        const quantitativeValData = engineeredValFeatures.map(row => 
            [...SCHEMA.quantitative, ...Object.keys(SCHEMA.derivedFeatures)].map(feature => row[feature])
        );
        
        const standardizedValData = applyStandardization(quantitativeValData, means, stds);
        
        // Combine train and validation data
        const allFeatures = [...standardizedTrainData, ...standardizedValData];
        const allTargets = [...trainTargets, ...valTargets];
        
        appState.processedData = {
            features: allFeatures,
            targets: allTargets,
            featureNames: [
                ...SCHEMA.quantitative,
                ...Object.keys(SCHEMA.derivedFeatures)
            ],
            trainIndices: trainIndices,
            valIndices: valIndices
        };
        
        showPreprocessingDetails();
        showStatus(elements.preprocessStatus(), 
            `✅ Advanced preprocessing complete! ${standardizedTrainData[0].length} engineered features`, 'success');
        
        elements.featureInfo().innerHTML = `
            <strong>🔧 Feature Engineering:</strong>
            <div style="margin-top: 8px;">
                • Original features: ${SCHEMA.features.length}<br>
                • Engineered features: ${Object.keys(SCHEMA.derivedFeatures).length}<br>
                • Total features: ${standardizedTrainData[0].length}<br>
                • Training samples: ${trainIndices.length}<br>
                • Validation samples: ${valIndices.length}
            </div>
        `;
        
        updateUIState();
        
    } catch (error) {
        showStatus(elements.preprocessStatus(), `❌ Preprocessing error: ${error.message}`, 'error');
        console.error('Preprocessing error:', error);
    }
}

function imputeMissingValues(features, useGlobalStats = false) {
    const processed = JSON.parse(JSON.stringify(features));
    
    SCHEMA.quantitative.forEach(feature => {
        let mean;
        
        if (useGlobalStats && appState.preprocessingStats.standardization[feature]) {
            // Use mean from training data
            mean = appState.preprocessingStats.standardization[feature].mean;
        } else {
            // Calculate mean from current data
            const values = processed.map(row => row[feature]).filter(val => val != null);
            mean = values.reduce((a, b) => a + b, 0) / values.length;
        }
        
        processed.forEach(row => {
            if (row[feature] == null) {
                row[feature] = mean;
            }
        });
    });
    
    return processed;
}

function engineerNewFeatures(features) {
    return features.map(row => {
        const newRow = { ...row };
        Object.entries(SCHEMA.derivedFeatures).forEach(([name, fn]) => {
            newRow[name] = fn(row);
        });
        return newRow;
    });
}

function standardizeDataWithStats(data) {
    const means = [];
    const stds = [];
    
    const transposed = data[0].map((_, colIndex) => data.map(row => row[colIndex]));
    
    const standardizedTransposed = transposed.map((column, index) => {
        const validValues = column.filter(val => !isNaN(val));
        const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
        const std = Math.sqrt(validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validValues.length);
        
        means[index] = mean;
        stds[index] = std || 1;
        
        return column.map(val => isNaN(val) ? 0 : (val - mean) / std);
    });
    
    const standardizedData = standardizedTransposed[0].map((_, rowIndex) => 
        standardizedTransposed.map(column => column[rowIndex])
    );
    
    return { standardizedData, means, stds };
}

function applyStandardization(data, means, stds) {
    return data.map(row => {
        return row.map((value, index) => {
            const mean = means[index];
            const std = stds[index];
            return isNaN(value) ? 0 : (value - mean) / std;
        });
    });
}

function showPreprocessingDetails() {
    if (!appState.preprocessingStats) return;
    
    let details = `<strong>📊 Preprocessing Details:</strong><br>`;
    details += `• Features engineered: ${Object.keys(SCHEMA.derivedFeatures).length}<br>`;
    details += `• Training samples: ${appState.trainIndices.length}<br>`;
    details += `• Validation samples: ${appState.valIndices.length}<br>`;
    details += `• Standardization applied to ${SCHEMA.quantitative.length} features<br>`;
    
    if (Object.keys(appState.preprocessingStats.standardization).length > 0) {
        details += `<br><strong>Standardization Parameters (from training data):</strong><br>`;
        Object.entries(appState.preprocessingStats.standardization).slice(0, 3).forEach(([feature, stats]) => {
            details += `• ${feature}: mean=${stats.mean.toFixed(2)}, std=${stats.std.toFixed(2)}<br>`;
        });
        if (Object.keys(appState.preprocessingStats.standardization).length > 3) {
            details += `• ... and ${Object.keys(appState.preprocessingStats.standardization).length - 3} more features`;
        }
    }
    
    elements.preprocessingDetails().innerHTML = details;
}

// ==================== MODEL FUNCTIONS ====================

function createModel() {
    try {
        showStatus(elements.modelStatus(), 'Creating neural network architecture...', 'loading');
        
        appState.model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [appState.processedData.featureNames.length],
                    units: 12,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid'
                })
            ]
        });
        
        appState.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        const totalParams = appState.model.countParams();
        
        elements.modelSummary().innerHTML = `
            <strong>🧠 Neural Network Architecture:</strong>
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.9rem; margin-top: 10px;">
                <strong>Model Layers:</strong><br>
                • Input: ${appState.processedData.featureNames.length} features<br>
                • Hidden: Dense(12 units, ReLU activation)<br>
                • Dropout: 30% rate<br>
                • Output: Dense(1 unit, Sigmoid activation)<br>
                • Total Parameters: ${totalParams.toLocaleString()}<br><br>
                <strong>Training Configuration:</strong><br>
                • Optimizer: Adam (learning rate: 0.001)<br>
                • Loss: Binary Crossentropy<br>
                • Metrics: Accuracy
            </div>
        `;
        
        showStatus(elements.modelStatus(), '✅ Neural network created successfully!', 'success');
        updateUIState();
        
    } catch (error) {
        showStatus(elements.modelStatus(), `❌ Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
}

async function loadModel() {
    try {
        const modelFile = document.getElementById('modelFile').files[0];
        if (!modelFile) {
            alert('Please select a model file to load');
            return;
        }

        showStatus(elements.modelStatus(), 'Loading model...', 'loading');

        // Load the model using TensorFlow.js
        appState.model = await tf.loadLayersModel(URL.createObjectURL(modelFile));
        
        // Recompile the model
        appState.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        const totalParams = appState.model.countParams();
        
        elements.modelSummary().innerHTML = `
            <strong>🧠 Loaded Neural Network Architecture:</strong>
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.9rem; margin-top: 10px;">
                <strong>Model successfully loaded!</strong><br>
                • Total Parameters: ${totalParams.toLocaleString()}<br><br>
                <strong>Model Configuration:</strong><br>
                • Optimizer: Adam (learning rate: 0.001)<br>
                • Loss: Binary Crossentropy<br>
                • Metrics: Accuracy
            </div>
        `;
        
        showStatus(elements.modelStatus(), '✅ Model loaded successfully!', 'success');
        updateUIState();
        
    } catch (error) {
        showStatus(elements.modelStatus(), `❌ Model loading error: ${error.message}`, 'error');
        console.error('Model loading error:', error);
    }
}

async function trainModel() {
    try {
        showStatus(elements.trainingStatus(), 'Training neural network...', 'loading');
        
        // Extract training and validation data
        const trainFeatures = appState.processedData.features
            .filter((_, index) => index < appState.trainIndices.length);
        const trainTargets = appState.processedData.targets
            .filter((_, index) => index < appState.trainIndices.length);
        
        const valFeatures = appState.processedData.features
            .filter((_, index) => index >= appState.trainIndices.length);
        const valTargets = appState.processedData.targets
            .filter((_, index) => index >= appState.trainIndices.length);
        
        const featuresTensor = tf.tensor2d(trainFeatures);
        const targetsTensor = tf.tensor1d(trainTargets);
        const valFeaturesTensor = tf.tensor2d(valFeatures);
        const valTargetsTensor = tf.tensor1d(valTargets);
        
        const batchSize = 32;
        const epochs = 20;
        
        elements.trainingProgress().innerHTML = '<div style="margin: 10px 0;">Training in progress... Please wait.</div>';
        
        const history = await appState.model.fit(featuresTensor, targetsTensor, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [valFeaturesTensor, valTargetsTensor],
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (epoch === epochs - 1) {
                        // Store the final metrics from training
                        appState.finalTrainingMetrics = {
                            loss: logs.loss.toFixed(4),
                            accuracy: (logs.acc * 100).toFixed(1),
                            val_loss: logs.val_loss.toFixed(4),
                            val_accuracy: (logs.val_acc * 100).toFixed(1)
                        };
                        
                        elements.trainingProgress().innerHTML = `
                            <div class="training-results">
                                <strong>Training Complete!</strong><br>
                                Final Loss: ${logs.loss.toFixed(4)}<br>
                                Final Accuracy: ${(logs.acc * 100).toFixed(1)}%<br>
                                Validation Loss: ${logs.val_loss.toFixed(4)}<br>
                                Validation Accuracy: ${(logs.val_acc * 100).toFixed(1)}%
                            </div>
                        `;
                    }
                }
            }
        });
        
        appState.trainingHistory = history;
        
        // Use validation data for metrics calculation
        const predictions = appState.model.predict(valFeaturesTensor);
        const predictedValues = await predictions.data();
        
        appState.validationPredictions = Array.from(predictedValues);
        appState.validationLabels = Array.from(await valTargetsTensor.data());
        
        // Calculate metrics with default threshold
        calculateAndDisplayMetrics();
        calculateFeatureImportance();
        
        showStatus(elements.trainingStatus(), 
            `✅ Training complete!`, 
            'success');
        
        updateUIState();
        
        tf.dispose([featuresTensor, targetsTensor, valFeaturesTensor, valTargetsTensor, predictions]);
        
    } catch (error) {
        showStatus(elements.trainingStatus(), `❌ Training error: ${error.message}`, 'error');
        console.error('Training error:', error);
    }
}

function calculateAndDisplayMetrics() {
    if (!appState.validationPredictions || !appState.validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    const binaryPredictions = appState.validationPredictions.map(p => p >= threshold ? 1 : 0);
    const actualLabels = appState.validationLabels;
    
    let truePositives = 0, falsePositives = 0, trueNegatives = 0, falseNegatives = 0;
    
    for (let i = 0; i < binaryPredictions.length; i++) {
        const predicted = binaryPredictions[i];
        const actual = actualLabels[i];
        
        if (predicted === 1 && actual === 1) truePositives++;
        else if (predicted === 1 && actual === 0) falsePositives++;
        else if (predicted === 0 && actual === 0) trueNegatives++;
        else if (predicted === 0 && actual === 1) falseNegatives++;
    }
    
    const accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);
    const precision = truePositives / (truePositives + falsePositives) || 0;
    const recall = truePositives / (truePositives + falseNegatives) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    // Store current metrics
    appState.currentMetrics = {
        accuracy: accuracy,
        precision: precision,
        recall: recall,
        f1: f1,
        threshold: threshold
    };
    
    // Update display with consistent metrics
    document.getElementById('accuracy').textContent = accuracy.toFixed(3);
    document.getElementById('precision').textContent = precision.toFixed(3);
    document.getElementById('recall').textContent = recall.toFixed(3);
    document.getElementById('f1').textContent = f1.toFixed(3);
}

function calculateFeatureImportance() {
    if (!appState.model || !appState.processedData) return;
    
    const importanceScores = [];
    const numFeatures = appState.processedData.featureNames.length;
    
    // Calculate correlation-based importance using training data only
    const trainFeatures = appState.processedData.features
        .filter((_, index) => index < appState.trainIndices.length);
    const trainTargets = appState.processedData.targets
        .filter((_, index) => index < appState.trainIndices.length);
    
    for (let i = 0; i < numFeatures; i++) {
        const featureValues = trainFeatures.map(row => row[i]);
        
        // Calculate correlation with target
        const featureMean = featureValues.reduce((a, b) => a + b, 0) / featureValues.length;
        const targetMean = trainTargets.reduce((a, b) => a + b, 0) / trainTargets.length;
        
        let numerator = 0;
        let denomX = 0;
        let denomY = 0;
        
        for (let j = 0; j < featureValues.length; j++) {
            numerator += (featureValues[j] - featureMean) * (trainTargets[j] - targetMean);
            denomX += Math.pow(featureValues[j] - featureMean, 2);
            denomY += Math.pow(trainTargets[j] - targetMean, 2);
        }
        
        const correlation = Math.abs(numerator / (Math.sqrt(denomX) * Math.sqrt(denomY))) || 0;
        
        importanceScores.push({
            feature: appState.processedData.featureNames[i],
            importance: correlation
        });
    }
    
    // Normalize importance scores
    const maxImportance = Math.max(...importanceScores.map(item => item.importance));
    importanceScores.forEach(item => {
        item.importance = maxImportance > 0 ? item.importance / maxImportance : 0;
    });
    
    importanceScores.sort((a, b) => b.importance - a.importance);
    appState.featureImportance = importanceScores;
    
    displayFeatureImportance();
}

function displayFeatureImportance() {
    if (!appState.featureImportance) return;
    
    let html = `<strong>🎯 Feature Importance Analysis:</strong><br><br>`;
    
    appState.featureImportance.slice(0, 8).forEach(item => {
        const percentage = (item.importance * 100).toFixed(1);
        html += `
            <div style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>${item.feature}</span>
                    <span style="font-weight: bold;">${percentage}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    });
    
    elements.featureImportance().innerHTML = html;
}

function updateThreshold(value) {
    elements.thresholdValue().textContent = value;
    if (appState.validationPredictions && appState.validationLabels) {
        calculateAndDisplayMetrics();
    }
}

async function predictTestData() {
    try {
        if (appState.testData.length === 0) {
            alert('No test data loaded. Please load test data first.');
            return;
        }
        
        showStatus(elements.predictionStatus(), 'Processing test data and generating predictions...', 'loading');
        
        const testFeatures = appState.testData.map(row => {
            const featureRow = {};
            SCHEMA.features.forEach(feature => {
                featureRow[feature] = row[feature];
            });
            return featureRow;
        });
        
        const processedTestFeatures = imputeMissingValues(testFeatures, true);
        const engineeredTestFeatures = engineerNewFeatures(processedTestFeatures);
        
        const quantitativeTestData = engineeredTestFeatures.map(row => 
            [...SCHEMA.quantitative, ...Object.keys(SCHEMA.derivedFeatures)].map(feature => row[feature])
        );
        
        const standardizedTestData = standardizeTestData(quantitativeTestData);
        
        const testTensor = tf.tensor2d(standardizedTestData);
        const predictions = appState.model.predict(testTensor);
        const predictionValues = await predictions.data();
        
        appState.testPredictions = Array.from(predictionValues).map((prob, index) => ({
            identifier: appState.testData[index][SCHEMA.identifier] || `Sample_${index + 1}`,
            probability: prob,
            prediction: prob >= parseFloat(document.getElementById('thresholdSlider').value) ? 1 : 0
        }));
        
        displayPredictions();
        showStatus(elements.predictionStatus(), 
            `✅ Predictions generated! ${appState.testPredictions.length} samples processed`, 
            'success');
        
        updateUIState();
        
        tf.dispose([testTensor, predictions]);
        
    } catch (error) {
        showStatus(elements.predictionStatus(), `❌ Prediction error: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

function standardizeTestData(testData) {
    if (!appState.preprocessingStats?.standardization) {
        return testData;
    }
    
    const means = SCHEMA.quantitative.map(feature => 
        appState.preprocessingStats.standardization[feature]?.mean || 0
    );
    const stds = SCHEMA.quantitative.map(feature => 
        appState.preprocessingStats.standardization[feature]?.std || 1
    );
    
    return testData.map(row => {
        return row.map((value, index) => {
            if (index < means.length) {
                const mean = means[index];
                const std = stds[index];
                return isNaN(value) ? 0 : (value - mean) / std;
            }
            return value;
        });
    });
}

function displayPredictions() {
    if (!appState.testPredictions) return;
    
    let html = `<strong>📊 Prediction Results:</strong><br><br>`;
    html += `<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
        <thead>
            <tr style="background: #f8f9fa;">
                <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Applicant</th>
                <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Approval Probability</th>
                <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Decision</th>
            </tr>
        </thead>
        <tbody>`;
    
    appState.testPredictions.slice(0, 20).forEach(pred => {
        const probabilityPercent = (pred.probability * 100).toFixed(1);
        const decision = pred.prediction === 1 ? 
            `<span style="color: #27ae60; font-weight: bold;">APPROVE</span>` : 
            `<span style="color: #e74c3c; font-weight: bold;">REJECT</span>`;
        
        html += `
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">${pred.identifier}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">
                    <div class="progress-bar" style="margin: 4px 0;">
                        <div class="progress-fill" style="width: ${probabilityPercent}%"></div>
                    </div>
                    ${probabilityPercent}%
                </td>
                <td style="padding: 8px; border: 1px solid #ddd;">${decision}</td>
            </tr>
        `;
    });
    
    html += `</tbody></table>`;
    
    if (appState.testPredictions.length > 20) {
        html += `<div style="margin-top: 10px; font-style: italic;">
            ... and ${appState.testPredictions.length - 20} more predictions
        </div>`;
    }
    
    elements.predictionResults().innerHTML = html;
}

function downloadPredictions() {
    if (!appState.testPredictions) return;
    
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    
    let csvContent = `${SCHEMA.identifier},Approval_Probability,Decision\n`;
    
    appState.testPredictions.forEach(pred => {
        const decision = pred.probability >= threshold ? 'APPROVE' : 'REJECT';
        csvContent += `${pred.identifier},${(pred.probability * 100).toFixed(2)}%,${decision}\n`;
    });
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'loan_predictions.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

async function exportModel() {
    try {
        if (!appState.model) {
            alert('No model to export');
            return;
        }

        showStatus(elements.modelStatus(), 'Exporting model...', 'loading');
        
        // Save the entire model (architecture + weights)
        await appState.model.save('downloads://loan_approval_model');
        
        showStatus(elements.modelStatus(), '✅ Model exported successfully!', 'success');
    } catch (error) {
        showStatus(elements.modelStatus(), `❌ Export error: ${error.message}`, 'error');
        console.error('Export error:', error);
    }
}

// ==================== EVENT HANDLERS ====================

async function handleFileUpload(event, isTestData = false) {
    try {
        const file = event.target.files[0];
        if (!file) return;
        
        const fileContent = await readFile(file);
        const data = parseCSV(fileContent);
        
        if (!isTestData) {
            validateSchema(data);
            appState.rawData = data;
            showStatus(elements.dataStatus(), `✅ Training data loaded! ${data.length} samples`, 'success');
            performComprehensiveEDA();
        } else {
            appState.testData = data;
            showStatus(elements.predictionStatus(), `✅ Test data loaded! ${data.length} samples`, 'success');
        }
        
        updateUIState();
        
    } catch (error) {
        if (!isTestData) {
            showStatus(elements.dataStatus(), `❌ Data loading error: ${error.message}`, 'error');
        } else {
            showStatus(elements.predictionStatus(), `❌ Test data error: ${error.message}`, 'error');
        }
        console.error('File upload error:', error);
    }
}

// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize TensorFlow.js
    tf.setBackend('cpu').then(() => {
        console.log('TensorFlow.js backend initialized');
    });
    
    // Set up event listeners
    document.getElementById('fileInput').addEventListener('change', (e) => handleFileUpload(e, false));
    document.getElementById('testFileInput').addEventListener('change', (e) => handleFileUpload(e, true));
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    document.getElementById('createModelBtn').addEventListener('click', createModel);
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    document.getElementById('predictBtn').addEventListener('click', predictTestData);
    document.getElementById('exportBtn').addEventListener('click', exportModel);
    document.getElementById('loadModelBtn').addEventListener('click', loadModel);
    document.getElementById('downloadPredictionsBtn').addEventListener('click', downloadPredictions);
    document.getElementById('thresholdSlider').addEventListener('input', (e) => updateThreshold(e.target.value));
    
    // Initialize threshold display
    updateThreshold(document.getElementById('thresholdSlider').value);
    
    // Initialize UI state
    updateUIState();
    
    console.log('Loan Approval AI System initialized successfully!');
});