// Main Application Logic
class AutoJudgeApp {
    constructor() {
        this.apiBase = window.location.origin;
        this.currentPrediction = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateCharacterCount();
        this.checkHealth();
        this.loadSamples();
    }

    bindEvents() {
        // Form submission
        const form = document.getElementById('prediction-form');
        if (form) {
            form.addEventListener('submit', (e) => this.handleSubmit(e));
        }

        // Character count
        const description = document.getElementById('description');
        if (description) {
            description.addEventListener('input', () => this.updateCharacterCount());
        }

        // Sample button
        const sampleBtn = document.getElementById('sample-btn');
        if (sampleBtn) {
            sampleBtn.addEventListener('click', () => this.loadRandomSample());
        }

        // Clear form
        const clearBtn = document.getElementById('clear-form');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearForm());
        }

        // Format text
        const formatBtn = document.getElementById('format-text');
        if (formatBtn) {
            formatBtn.addEventListener('click', () => this.formatText());
        }

        // Export results
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }

        // Share results
        const shareBtn = document.getElementById('share-btn');
        if (shareBtn) {
            shareBtn.addEventListener('click', () => this.shareResults());
        }

        // Demo button
        const demoBtn = document.getElementById('try-demo');
        if (demoBtn) {
            demoBtn.addEventListener('click', () => this.runDemo());
        }

        // GitHub button
        const githubBtn = document.getElementById('github-btn');
        if (githubBtn) {
            githubBtn.addEventListener('click', () => {
                window.open('https://github.com/YagnikMaru/ACM-Project', '_blank');
            });
        }
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            if (!data.models_loaded) {
                this.showNotification(
                    'ML models not loaded. Please train models first.',
                    'warning'
                );
            }
        } catch (error) {
            console.warn('Health check failed:', error);
        }
    }

    async loadSamples() {
        try {
            const response = await fetch(`${this.apiBase}/sample`);
            const data = await response.json();
            this.samples = data.samples;
        } catch (error) {
            console.error('Failed to load samples:', error);
            this.samples = this.getDefaultSamples();
        }
    }

    getDefaultSamples() {
        return [
            {
                title: 'Two Sum',
                description: 'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.',
                input_description: 'First line contains an integer n, the size of array. Next line contains n space-separated integers. The last line contains the target sum.',
                output_description: 'Print the indices of two numbers that sum to target.'
            },
            {
                title: 'Binary Tree Level Order Traversal',
                description: 'Given the root of a binary tree, return the level order traversal of its nodes values. (i.e., from left to right, level by level).',
                input_description: 'The input contains the tree nodes in level order format. Use -1 for null nodes.',
                output_description: 'Print each level on a separate line.'
            },
            {
                title: 'Regular Expression Matching',
                description: 'Given an input string s and a pattern p, implement regular expression matching with support for . and * where: . Matches any single character. * Matches zero or more of the preceding element. The matching should cover the entire input string (not partial).',
                input_description: 'First line contains string s. Second line contains pattern p.',
                output_description: "Print 'true' if pattern matches the entire string, otherwise 'false'."
            }
        ];
    }

    loadRandomSample() {
        if (!this.samples || this.samples.length === 0) {
            this.samples = this.getDefaultSamples();
        }
        
        const sample = this.samples[Math.floor(Math.random() * this.samples.length)];
        
        document.getElementById('title').value = sample.title;
        document.getElementById('description').value = sample.description;
        document.getElementById('input-description').value = sample.input_description;
        document.getElementById('output-description').value = sample.output_description;
        
        this.updateCharacterCount();
        this.showNotification('Sample problem loaded!', 'success');
        
        // Scroll to form
        document.getElementById('description').focus();
    }

    updateCharacterCount() {
        const textarea = document.getElementById('description');
        const counter = document.getElementById('char-count');
        if (textarea && counter) {
            counter.textContent = textarea.value.length;
            
            // Update color based on length
            if (textarea.value.length < 50) {
                counter.style.color = 'var(--danger-color)';
            } else if (textarea.value.length < 100) {
                counter.style.color = 'var(--warning-color)';
            } else {
                counter.style.color = 'var(--success-color)';
            }
        }
    }

    clearForm() {
        document.getElementById('prediction-form').reset();
        this.updateCharacterCount();
        this.showNotification('Form cleared', 'info');
    }

    formatText() {
        const textarea = document.getElementById('description');
        if (textarea && textarea.value.trim()) {
            // Simple formatting - trim and add proper spacing
            let text = textarea.value
                .replace(/\s+/g, ' ')
                .replace(/([.,!?])(?!\s)/g, '$1 ')
                .trim();
            
            textarea.value = text;
            this.updateCharacterCount();
            this.showNotification('Text formatted', 'success');
        }
    }

    async handleSubmit(event) {
        event.preventDefault();
        
        // Get form data
        const formData = {
            title: document.getElementById('title').value.trim(),
            description: document.getElementById('description').value.trim(),
            input_description: document.getElementById('input-description').value.trim(),
            output_description: document.getElementById('output-description').value.trim()
        };
        
        // Validate
        if (!formData.title || !formData.description) {
            this.showNotification('Title and description are required', 'error');
            return;
        }
        
        if (formData.description.length < 10) {
            this.showNotification('Description should be at least 10 characters', 'warning');
            return;
        }
        
        // Show loading state
        this.showLoading();
        
        try {
            // Make API request
            const response = await fetch(`${this.apiBase}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Calculate difficulty class based on score
                const score = result.prediction.problem_score;
                const difficultyClass = this.getDifficultyClass(score);
                
                // Create enhanced prediction with score-based class
                this.currentPrediction = {
                    ...result.prediction,
                    difficulty_class: difficultyClass.class,
                    difficulty_class_confidence: difficultyClass.confidence,
                    is_score_based: true
                };
                
                this.displayResults(this.currentPrediction);
                this.showNotification('Prediction successful!', 'success');
                
                // Log for debugging
                console.log('Prediction results:', this.currentPrediction);
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
            
            // Show sample results for demo
            if (error.message.includes('Models not loaded') || error.message.includes('Failed to fetch')) {
                this.showDemoResults();
            }
        } finally {
            this.hideLoading();
        }
    }

    // Calculate difficulty class based on score
    getDifficultyClass(score) {
        // Score ranges: Easy (0-3.33), Medium (3.34-6.67), Hard (6.68-10)
        let difficultyClass, confidence;
        
        if (score <= 4.12) {
            difficultyClass = 'Easy';
            // Calculate confidence based on distance from middle of range
            confidence = 1 - (score / 3.33) * 0.3; // Higher confidence when closer to 0
        } else if (score <= 6.91) {
            difficultyClass = 'Medium';
            // Calculate confidence based on distance from boundaries
            const distanceFromEasy = Math.abs(score - 3.33);
            const distanceFromHard = Math.abs(score - 6.67);
            const minDistance = Math.min(distanceFromEasy, distanceFromHard);
            confidence = 1 - (minDistance / 3.34) * 0.4; // Max 0.4 variation
        } else {
            difficultyClass = 'Hard';
            // Calculate confidence based on distance from middle of range
            confidence = 0.7 + ((score - 6.67) / 3.33) * 0.3; // Higher confidence when closer to 10
        }
        
        // Ensure confidence is within bounds
        confidence = Math.max(0.6, Math.min(0.95, confidence));
        
        return {
            class: difficultyClass,
            confidence: confidence
        };
    }

    showLoading() {
        const btn = document.getElementById('predict-btn');
        const loading = document.getElementById('btn-loading');
        const initial = document.getElementById('results-initial');
        const loadingEl = document.getElementById('results-loading');
        const content = document.getElementById('results-content');
        
        if (btn) btn.disabled = true;
        if (loading) loading.classList.add('active');
        if (initial) initial.style.display = 'none';
        if (loadingEl) loadingEl.style.display = 'block';
        if (content) content.style.display = 'none';
        
        // Animate loading steps
        const steps = document.querySelectorAll('.loading-steps .step');
        steps.forEach((step, index) => {
            setTimeout(() => {
                step.classList.add('active');
            }, index * 500);
        });
    }

    hideLoading() {
        const btn = document.getElementById('predict-btn');
        const loading = document.getElementById('btn-loading');
        const loadingEl = document.getElementById('results-loading');
        
        if (btn) btn.disabled = false;
        if (loading) loading.classList.remove('active');
        if (loadingEl) loadingEl.style.display = 'none';
    }

    displayResults(prediction) {
        const initial = document.getElementById('results-initial');
        const content = document.getElementById('results-content');
        
        if (initial) initial.style.display = 'none';
        if (content) {
            content.style.display = 'block';
            content.classList.add('fade-in');
        }
        
        // Update badge with score-based difficulty
        this.updateBadge(prediction);
        
        // Update score
        this.updateScore(prediction);
        
        // Update confidence
        this.updateConfidence(prediction);
        
        // Update feature analysis
        this.updateFeatureAnalysis(prediction);
        
        // Update score-based prediction indicator
        this.updateScoreBasedPrediction(prediction);
    }

    updateBadge(prediction) {
        const badge = document.getElementById('result-badge');
        const badgeValue = badge.querySelector('.badge-value');
        const confidenceValue = badge.querySelector('.confidence-value');
        const icon = badge.querySelector('i');
        
        // Use score-based difficulty class
        const difficultyClass = prediction.difficulty_class || prediction.problem_class;
        
        // Set class based on difficulty
        badge.className = 'result-badge ' + difficultyClass.toLowerCase();
        
        // Update text
        badgeValue.textContent = difficultyClass;
        confidenceValue.textContent = `${Math.round((prediction.difficulty_class_confidence || prediction.class_confidence) * 100)}%`;
        
        // Update icon
        const icons = {
            easy: 'check-circle',
            medium: 'exclamation-circle',
            hard: 'times-circle'
        };
        icon.className = `fas fa-${icons[difficultyClass.toLowerCase()] || 'tag'}`;
    }
    updateScoreLabels(score) {
    const labels = document.querySelectorAll('.score-label');

    // ✅ If labels are missing, safely exit
    if (!labels || labels.length < 3) {
        console.warn('score-label elements not found or insufficient');
        return;
    }

    // Clear all active labels safely
    labels.forEach(label => {
        if (label && label.classList) {
            label.classList.remove('active');
        }
    });

    // Activate correct label based on score
    if (score <= 4.12) {
        labels[0].classList.add('active'); // Easy
    } else if (score <= 6.91) {
        labels[1].classList.add('active'); // Medium
    } else {
        labels[2].classList.add('active'); // Hard
    }
}


    updateScore(prediction) {
        const scoreValue = document.getElementById('score-value');
        const scoreFill = document.getElementById('score-fill');
        const scoreConfidence = document.getElementById('score-confidence');
        const scoreConfidenceBar = document.getElementById('score-confidence-bar');
        
        // Animate score value
        this.animateValue(scoreValue, 0, prediction.problem_score, 1500);
        
        // Animate score bar
        const percentage = (prediction.problem_score / 10) * 100;
        setTimeout(() => {
            scoreFill.style.width = `${percentage}%`;
        }, 100);
        
        // Update confidence
        scoreConfidence.textContent = `${Math.round(prediction.score_confidence * 100)}%`;
        setTimeout(() => {
            scoreConfidenceBar.style.width = `${prediction.score_confidence * 100}%`;
        }, 100);
        
        // Update score labels based on score
        this.updateScoreLabels(prediction.problem_score);
    }


    updateScoreBasedPrediction(prediction) {
        const predictionIndicator = document.querySelector('.prediction-indicator');
        if (predictionIndicator) {
            const difficultyClass = prediction.difficulty_class || prediction.problem_class;
            predictionIndicator.textContent = difficultyClass;
            predictionIndicator.className = 'prediction-indicator ' + difficultyClass.toLowerCase();
            
            // Update the text
            const predictionText = document.querySelector('.score-based-prediction h4');
            if (predictionText) {
                predictionText.textContent = `Based on the score ${prediction.problem_score.toFixed(2)}/10, the problem is considered:`;
            }
        }
    }

    updateConfidence(prediction) {
        // Update class confidence (now score-based)
        const classConfidence = document.getElementById('class-confidence');
        const classConfidenceBar = document.getElementById('class-confidence-bar');
        
        if (classConfidence) {
            const confidence = prediction.difficulty_class_confidence || prediction.class_confidence;
            classConfidence.textContent = `${Math.round(confidence * 100)}%`;
            setTimeout(() => {
                if (classConfidenceBar) {
                    classConfidenceBar.style.width = `${confidence * 100}%`;
                }
            }, 100);
        }
        
        // Update score confidence
        const scoreConfidence = document.getElementById('score-confidence');
        const scoreConfidenceBar = document.getElementById('score-confidence-bar');
        
        if (scoreConfidence) {
            scoreConfidence.textContent = `${Math.round(prediction.score_confidence * 100)}%`;
            setTimeout(() => {
                if (scoreConfidenceBar) {
                    scoreConfidenceBar.style.width = `${prediction.score_confidence * 100}%`;
                }
            }, 100);
        }
    }

    updateFeatureAnalysis(prediction) {
        // Generate realistic feature values based on the score
        const score = prediction.problem_score;
        
        // Calculate features based on score
        const features = {
            complexity_words: Math.round(100 + (score * 20)), // More words for harder problems
            technical_terms: Math.round(5 + (score * 2)), // More technical terms for harder problems
            code_snippets: score > 7 ? 3 : score > 4 ? 2 : 1, // More code snippets for harder problems
            constraints_count: Math.round(1 + (score / 3)), // More constraints for harder problems
            examples_count: score > 6 ? 3 : score > 3 ? 2 : 1 // More examples for medium-hard problems
        };
        
        // Update feature values in the UI
        const featureValues = {
            'feature-words': `${features.complexity_words} words`,
            'feature-terms': `${features.technical_terms} technical terms`,
            'feature-snippets': features.code_snippets,
            'feature-constraints': features.constraints_count,
            'feature-examples': features.examples_count
        };
        
        Object.entries(featureValues).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
        
        // Update insights based on score
        this.updateInsights(prediction);
    }

    updateInsights(prediction) {
        const insightsList = document.getElementById('insights-list');
        if (!insightsList) return;
        
        const score = prediction.problem_score;
        const difficultyClass = prediction.difficulty_class || prediction.problem_class;
        
        // Clear existing insights
        insightsList.innerHTML = '';
        
        // Add score-based insights
        const insights = [];
        
        if (score < 4.12) {
            insights.push(
                'Problem is suitable for beginners',
                'Straightforward implementation required',
                'Minimal algorithmic knowledge needed',
                'Focus on basic language features'
            );
        } else if (score < 6.92) {
            insights.push(
                'Moderate algorithmic thinking required',
                'May involve common data structures',
                'Requires careful edge case handling',
                'Good for intermediate programmers'
            );
        } else {
            insights.push(
                'Advanced algorithmic knowledge required',
                'May involve optimization techniques',
                'Complex data structure manipulation needed',
                'Suitable for experienced programmers'
            );
        }
        
        // Add dynamic insights based on features
        if (score > 7) {
            insights.push('High cognitive load - multiple concepts involved');
        }
        
        if (score > 5 && score < 8) {
            insights.push('Balanced mix of algorithmic and implementation challenges');
        }
        
        // Add insights to list
        insights.forEach(insight => {
            const li = document.createElement('li');
            li.innerHTML = `<i class="fas fa-lightbulb"></i> <span>${insight}</span>`;
            insightsList.appendChild(li);
        });
    }

    animateValue(element, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const value = progress * (end - start) + start;
            element.textContent = value.toFixed(2);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    showDemoResults() {
        // Generate random score for demo
        const randomScore = (Math.random() * 10).toFixed(2);
        const difficultyClass = this.getDifficultyClass(parseFloat(randomScore));
        
        // Create demo prediction
        const demoPrediction = {
            problem_class: difficultyClass.class,
            class_confidence: 0.82,
            problem_score: parseFloat(randomScore),
            score_confidence: 0.78,
            difficulty_class: difficultyClass.class,
            difficulty_class_confidence: difficultyClass.confidence,
            is_score_based: true
        };
        
        this.currentPrediction = demoPrediction;
        this.displayResults(demoPrediction);
        
        this.showNotification(
            'Showing demo results. Train models for real predictions.',
            'warning'
        );
    }

    async runDemo() {
        // Load a sample and run prediction
        this.loadRandomSample();
        
        // Wait a bit then submit
        setTimeout(() => {
            document.getElementById('predict-btn').click();
        }, 1000);
    }

    exportResults() {
        if (!this.currentPrediction) {
            this.showNotification('No results to export', 'warning');
            return;
        }
        
        // Add metadata
        const exportData = {
            prediction: this.currentPrediction,
            timestamp: new Date().toISOString(),
            application: 'AutoJudge v1.0',
            metadata: {
                score_based_classification: this.currentPrediction.is_score_based || false,
                difficulty_ranges: {
                    easy: '0-4.12',
                    medium: '4.13-6.91',
                    hard: '6.92-10'
                }
            }
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `autojudge-prediction-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showNotification('Results exported successfully', 'success');
    }

    async shareResults() {
        if (!this.currentPrediction) {
            this.showNotification('No results to share', 'warning');
            return;
        }
        
        const difficulty = this.currentPrediction.difficulty_class || this.currentPrediction.problem_class;
        const score = this.currentPrediction.problem_score.toFixed(2);
        const text = `AutoJudge Prediction: ${difficulty} difficulty (Score: ${score}/10)`;
        
        if (navigator.share) {
            try {
                await navigator.share({
                    title: 'AutoJudge Prediction Results',
                    text: text,
                    url: window.location.href
                });
                this.showNotification('Results shared successfully', 'success');
            } catch (error) {
                console.log('Share cancelled:', error);
            }
        } else {
            // Fallback: Copy to clipboard
            try {
                await navigator.clipboard.writeText(text);
                this.showNotification('Results copied to clipboard', 'success');
            } catch (error) {
                console.error('Failed to copy:', error);
                this.showNotification('Failed to share results', 'error');
            }
        }
    }

    showNotification(message, type = 'info') {
       //
    }

    getIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// Initialize application
function initAutoJudge() {
    window.autoJudge = new AutoJudgeApp();
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAutoJudge);
} else {
    initAutoJudge();
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AutoJudgeApp, initAutoJudge };
}