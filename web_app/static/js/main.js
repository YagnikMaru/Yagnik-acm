/**
 * AutoJudge - Main JavaScript
 * Handles form submission, API calls, and result display
 */

// API Configuration
const API_URL = window.location.origin;

// State Management
let currentPrediction = null;

// Sample Problems
const SAMPLE_PROBLEMS = [
    {
        title: "Two Sum",
        description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.",
        input: "nums = [2,7,11,15], target = 9",
        output: "[0,1] (Because nums[0] + nums[1] == 9, we return [0, 1])"
    },
    {
        title: "Longest Increasing Subsequence",
        description: "Given an integer array nums, return the length of the longest strictly increasing subsequence. A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. Use dynamic programming with O(n log n) time complexity using binary search optimization.",
        input: "nums = [10,9,2,5,3,7,101,18]",
        output: "4 (The longest increasing subsequence is [2,3,7,101])"
    },
    {
        title: "Binary Tree Level Order Traversal",
        description: "Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level). Implement using BFS (Breadth-First Search) algorithm with a queue.",
        input: "root = [3,9,20,null,null,15,7]",
        output: "[[3],[9,20],[15,7]]"
    }
];

/**
 * Initialize AutoJudge functionality
 */
function initAutoJudge() {
    console.log('🚀 Initializing AutoJudge...');
    
    // Get form elements
    const form = document.getElementById('prediction-form');
    const titleInput = document.getElementById('title');
    const descriptionInput = document.getElementById('description');
    const inputDescInput = document.getElementById('input-description');
    const outputDescInput = document.getElementById('output-description');
    const charCount = document.getElementById('char-count');
    const predictBtn = document.getElementById('predict-btn');
    const btnLoading = document.getElementById('btn-loading');
    
    // Get result elements
    const resultsInitial = document.getElementById('results-initial');
    const resultsLoading = document.getElementById('results-loading');
    const resultsContent = document.getElementById('results-content');
    
    // Update character count
    if (descriptionInput && charCount) {
        descriptionInput.addEventListener('input', function() {
            charCount.textContent = this.value.length;
        });
    }
    
    // Clear form button
    const clearBtn = document.getElementById('clear-form');
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            form.reset();
            charCount.textContent = '0';
            showInitialState();
        });
    }
    
    // Sample button
    const sampleBtn = document.getElementById('sample-btn');
    if (sampleBtn) {
        sampleBtn.addEventListener('click', loadSampleProblem);
    }
    
    // Format text button
    const formatBtn = document.getElementById('format-text');
    if (formatBtn) {
        formatBtn.addEventListener('click', function() {
            if (descriptionInput.value) {
                descriptionInput.value = descriptionInput.value.trim().replace(/\s+/g, ' ');
                charCount.textContent = descriptionInput.value.length;
            }
        });
    }
    
    // Export button
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportResults);
    }
    
    // Share button
    const shareBtn = document.getElementById('share-btn');
    if (shareBtn) {
        shareBtn.addEventListener('click', shareResults);
    }
    
    // GitHub button
    const githubBtn = document.getElementById('github-btn');
    if (githubBtn) {
        githubBtn.addEventListener('click', function() {
            window.open('https://github.com/YagnikMaru/AutoJudge', '_blank');
        });
    }
    
    // Form submission
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    console.log('✅ AutoJudge initialized successfully');
}

/**
 * Load a random sample problem
 */
function loadSampleProblem() {
    const sample = SAMPLE_PROBLEMS[Math.floor(Math.random() * SAMPLE_PROBLEMS.length)];
    
    document.getElementById('title').value = sample.title;
    document.getElementById('description').value = sample.description;
    document.getElementById('input-description').value = sample.input;
    document.getElementById('output-description').value = sample.output;
    
    // Update character count
    document.getElementById('char-count').textContent = sample.description.length;
    
    // Show notification
    showNotification('Sample problem loaded!', 'success');
}

/**
 * Handle form submission
 */
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Get form data
    const title = document.getElementById('title').value.trim();
    const description = document.getElementById('description').value.trim();
    const input = document.getElementById('input-description').value.trim();
    const output = document.getElementById('output-description').value.trim();
    
    // Validation
    if (!title) {
        showNotification('Please enter a problem title', 'error');
        return;
    }
    
    if (!description) {
        showNotification('Please enter a problem description', 'error');
        return;
    }
    
    if (description.length < 20) {
        showNotification('Description is too short (minimum 20 characters)', 'error');
        return;
    }
    
    // Show loading state
    showLoadingState();
    
    try {
        // Call API
        const result = await predictDifficulty({
            title,
            description,
            input,
            output
        });
        
        // Show results
        displayResults(result);
        
        // Smooth scroll to results
        document.getElementById('results-content').scrollIntoView({
            behavior: 'smooth',
            block: 'nearest'
        });
        
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Prediction failed: ' + error.message, 'error');
        showInitialState();
    }
}

/**
 * Call prediction API
 */
async function predictDifficulty(data) {
    const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
    }
    
    return await response.json();
}

/**
 * Display prediction results
 */
function displayResults(data) {
    console.log('📊 Displaying results:', data);
    
    // Store current prediction
    currentPrediction = data;
    
    // Hide loading, show results
    document.getElementById('results-loading').style.display = 'none';
    document.getElementById('results-initial').style.display = 'none';
    document.getElementById('results-content').style.display = 'block';
    
    // Update difficulty badge
    const badge = document.getElementById('result-badge');
    const badgeValue = badge.querySelector('.badge-value');
    const confidenceValue = badge.querySelector('.confidence-value');
    
    badgeValue.textContent = data.problem_class;
    confidenceValue.textContent = Math.round(data.confidence * 100) + '%';
    
    // Set badge color based on difficulty
    badge.className = 'result-badge ' + data.problem_class.toLowerCase();
    
    // Update score
    const scoreValue = document.getElementById('score-value');
    const scoreFill = document.getElementById('score-fill');
    const scoreConfidence = document.getElementById('score-confidence');
    
    scoreValue.textContent = data.problem_score.toFixed(2);
    
    // Animate score bar
    const scorePercent = (data.problem_score * 10);
    if(data.problem_score <= 4){
        scoreValue.textContent = (data.problem_score*data.confidence).toFixed(2);
        badgeValue.textContent = "Easy"
    }
    if(data.problem_score >= 5.8){
        badgeValue.textContent = "Hard"
        scoreValue.textContent = (data.problem_score.toFixed(2)*(0.90 + data.confidence)).toFixed(2);
    }
    if(data.problem_score > 4 && data.problem_score < 5.8){
        badgeValue.textContent = "Medium"
    }
    setTimeout(() => {
        scoreFill.style.width = scorePercent + '%';
    }, 100);
    
    // Set score bar color
    if (data.problem_score < 4) {
        scoreFill.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
    } else if (data.problem_score < 5.8) {
        scoreFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
    } else if(data.problem_score.toFixed(2)*(0.90 + data.confidence) >= 6){
        scoreFill.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
    }
    
    // Update confidence
    const confidencePercent = Math.round(data.confidence * 100);
    scoreConfidence.textContent = confidencePercent + '%';
    
    // Update feature analysis
    updateFeatureAnalysis(data);
    
    // Show success notification
    showNotification('Prediction complete!', 'success');
}

/**
 * Update feature analysis display
 */
function updateFeatureAnalysis(data) {
    const featureGrid = document.querySelector('.feature-grid');
    if (!featureGrid) return;
    
    const metadata = data.metadata || {};
    
    featureGrid.innerHTML = `
        <div class="feature-item">
            <div class="feature-icon">
                <i class="fas fa-font"></i>
            </div>
            <div class="feature-info">
                <div class="feature-name">Text Length</div>
                <div class="feature-value">${metadata.word_count || 0} words</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">
                <i class="fas fa-cube"></i>
            </div>
            <div class="feature-info">
                <div class="feature-name">Features Used</div>
                <div class="feature-value">${metadata.features_used || 0}</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">
                <i class="fas fa-chart-line"></i>
            </div>
            <div class="feature-info">
                <div class="feature-name">Score Range</div>
                <div class="feature-value">[${data.theoretical_range ? data.theoretical_range.join(', ') : '0, 10'}]</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">
                <i class="fas fa-robot"></i>
            </div>
            <div class="feature-info">
                <div class="feature-name">Model</div>
                <div class="feature-value">${metadata.regressor_used ? 'Full' : 'Basic'}</div>
            </div>
        </div>
    `;
}

/**
 * Show initial state
 */
function showInitialState() {
    document.getElementById('results-initial').style.display = 'block';
    document.getElementById('results-loading').style.display = 'none';
    document.getElementById('results-content').style.display = 'none';
}

/**
 * Show loading state
 */
function showLoadingState() {
    document.getElementById('results-initial').style.display = 'none';
    document.getElementById('results-loading').style.display = 'block';
    document.getElementById('results-content').style.display = 'none';
    
    // Animate loading steps
    animateLoadingSteps();
}

/**
 * Animate loading steps
 */
function animateLoadingSteps() {
    const steps = document.querySelectorAll('.loading-steps .step');
    
    steps.forEach((step, index) => {
        setTimeout(() => {
            step.classList.add('active');
        }, index * 400);
    });
}

/**
 * Export results as JSON
 */
function exportResults() {
    if (!currentPrediction) {
        showNotification('No prediction to export', 'error');
        return;
    }
    
    const dataStr = JSON.stringify(currentPrediction, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `autojudge_prediction_${Date.now()}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
    showNotification('Results exported successfully!', 'success');
}

/**
 * Share results
 */
function shareResults() {
    if (!currentPrediction) {
        showNotification('No prediction to share', 'error');
        return;
    }
    
    const text = `AutoJudge Prediction:\n` +
                 `Difficulty: ${currentPrediction.problem_class}\n` +
                 `Score: ${currentPrediction.problem_score.toFixed(2)}/10\n` +
                 `Confidence: ${Math.round(currentPrediction.confidence * 100)}%`;
    
    if (navigator.share) {
        navigator.share({
            title: 'AutoJudge Prediction',
            text: text
        }).then(() => {
            showNotification('Shared successfully!', 'success');
        }).catch((error) => {
            console.log('Share failed:', error);
            copyToClipboard(text);
        });
    } else {
        copyToClipboard(text);
    }
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy to clipboard', 'error');
    });
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    //
}

// Export for global access
window.initAutoJudge = initAutoJudge;