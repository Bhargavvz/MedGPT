/**
 * Medical VQA - Frontend JavaScript
 * Handles image upload, API calls, and result display
 */

// ============================================================================
// Configuration
// ============================================================================

const API_BASE = window.location.origin;
const API_ENDPOINTS = {
    vqa: `${API_BASE}/api/vqa`,
    report: `${API_BASE}/api/report`,
    health: `${API_BASE}/api/health`,
    sampleQuestions: `${API_BASE}/api/sample-questions`,
};

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    // Theme
    themeToggle: document.getElementById('themeToggle'),

    // Upload
    uploadZone: document.getElementById('uploadZone'),
    imageInput: document.getElementById('imageInput'),
    imagePreview: document.getElementById('imagePreview'),
    previewImg: document.getElementById('previewImg'),
    removeImage: document.getElementById('removeImage'),

    // Question
    questionInput: document.getElementById('questionInput'),
    submitBtn: document.getElementById('submitBtn'),
    showExplanation: document.getElementById('showExplanation'),
    showHeatmap: document.getElementById('showHeatmap'),

    // Results
    resultsSection: document.getElementById('resultsSection'),
    answerText: document.getElementById('answerText'),
    explanationBlock: document.getElementById('explanationBlock'),
    explanationText: document.getElementById('explanationText'),
    heatmapBlock: document.getElementById('heatmapBlock'),
    heatmapImg: document.getElementById('heatmapImg'),
    processingTime: document.getElementById('processingTime'),
    newQuestionBtn: document.getElementById('newQuestionBtn'),
    downloadReportBtn: document.getElementById('downloadReportBtn'),

    // Loading
    loadingOverlay: document.getElementById('loadingOverlay'),
};

// ============================================================================
// State
// ============================================================================

let state = {
    uploadedFile: null,
    lastResult: null,
};

// ============================================================================
// Theme Toggle
// ============================================================================

function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
}

function toggleTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const newTheme = isDark ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
}

// ============================================================================
// File Upload
// ============================================================================

function initUpload() {
    const { uploadZone, imageInput, removeImage } = elements;

    // Click to upload
    uploadZone.addEventListener('click', () => imageInput.click());

    // File input change
    imageInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Remove image
    removeImage.addEventListener('click', (e) => {
        e.stopPropagation();
        clearUpload();
    });
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'application/dicom'];
    if (!validTypes.includes(file.type) && !file.name.endsWith('.dcm')) {
        showToast('Please upload a valid image file (JPEG, PNG, or DICOM)', 'error');
        return;
    }

    // Validate file size (50MB max)
    if (file.size > 50 * 1024 * 1024) {
        showToast('File size must be less than 50MB', 'error');
        return;
    }

    state.uploadedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImg.src = e.target.result;
        elements.uploadZone.hidden = true;
        elements.imagePreview.hidden = false;
        updateSubmitButton();
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    state.uploadedFile = null;
    elements.imageInput.value = '';
    elements.uploadZone.hidden = false;
    elements.imagePreview.hidden = true;
    updateSubmitButton();
}

// ============================================================================
// Question Input
// ============================================================================

function initQuestion() {
    const { questionInput, submitBtn } = elements;

    // Input listener
    questionInput.addEventListener('input', updateSubmitButton);

    // Enter to submit (with Ctrl/Cmd)
    questionInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            submitQuestion();
        }
    });

    // Submit button
    submitBtn.addEventListener('click', submitQuestion);

    // Sample questions
    document.querySelectorAll('.chip[data-question]').forEach(chip => {
        chip.addEventListener('click', () => {
            questionInput.value = chip.dataset.question;
            updateSubmitButton();
            questionInput.focus();
        });
    });

    // New question button
    elements.newQuestionBtn.addEventListener('click', () => {
        elements.resultsSection.hidden = true;
        questionInput.value = '';
        questionInput.focus();
    });

    // Download report button
    elements.downloadReportBtn.addEventListener('click', downloadReport);
}

function updateSubmitButton() {
    const hasFile = state.uploadedFile !== null;
    const hasQuestion = elements.questionInput.value.trim().length > 0;
    elements.submitBtn.disabled = !(hasFile && hasQuestion);
}

// ============================================================================
// API Calls
// ============================================================================

async function submitQuestion() {
    if (!state.uploadedFile || !elements.questionInput.value.trim()) {
        return;
    }

    const { questionInput, showExplanation, showHeatmap } = elements;

    // Prepare form data
    const formData = new FormData();
    formData.append('image', state.uploadedFile);
    formData.append('question', questionInput.value.trim());
    formData.append('generate_explanation', showExplanation.checked);
    formData.append('generate_heatmap', showHeatmap.checked);

    try {
        showLoading(true);

        const response = await fetch(API_ENDPOINTS.vqa, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }

        const result = await response.json();
        state.lastResult = result;
        displayResults(result);

    } catch (error) {
        console.error('VQA Error:', error);
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

function displayResults(result) {
    const {
        resultsSection,
        answerText,
        explanationBlock,
        explanationText,
        heatmapBlock,
        heatmapImg,
        processingTime,
    } = elements;

    // Set answer
    answerText.textContent = result.answer;

    // Set explanation
    if (result.explanation) {
        explanationText.textContent = result.explanation;
        explanationBlock.hidden = false;
    } else {
        explanationBlock.hidden = true;
    }

    // Set heatmap
    if (result.heatmap_base64) {
        heatmapImg.src = `data:image/png;base64,${result.heatmap_base64}`;
        heatmapBlock.hidden = false;
    } else {
        heatmapBlock.hidden = true;
    }

    // Set metadata
    processingTime.textContent = `Processing time: ${result.processing_time.toFixed(2)}s`;

    // Show results section
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function downloadReport() {
    if (!state.lastResult) return;

    const report = {
        timestamp: new Date().toISOString(),
        question: state.lastResult.question,
        answer: state.lastResult.answer,
        explanation: state.lastResult.explanation || 'N/A',
        processing_time: state.lastResult.processing_time,
        session_id: state.lastResult.session_id,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `medical_vqa_report_${Date.now()}.json`;
    a.click();

    URL.revokeObjectURL(url);
    showToast('Report downloaded successfully!', 'success');
}

// ============================================================================
// Health Check
// ============================================================================

async function checkHealth() {
    try {
        const response = await fetch(API_ENDPOINTS.health);
        const data = await response.json();

        if (data.status === 'healthy') {
            console.log('API is healthy:', data);
            if (!data.model_loaded) {
                showToast('Running in demo mode - model not loaded', 'warning');
            }
        }
    } catch (error) {
        console.warn('Health check failed:', error);
        showToast('API server may be unavailable', 'warning');
    }
}

// ============================================================================
// UI Helpers
// ============================================================================

function showLoading(show) {
    elements.loadingOverlay.hidden = !show;
    elements.submitBtn.disabled = show;

    const btnText = elements.submitBtn.querySelector('.btn-text');
    const btnLoader = elements.submitBtn.querySelector('.btn-loader');

    if (show) {
        btnText.hidden = true;
        btnLoader.hidden = false;
    } else {
        btnText.hidden = false;
        btnLoader.hidden = true;
    }
}

function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${getToastIcon(type)}</span>
        <span class="toast-message">${message}</span>
    `;

    // Add styles if not present
    if (!document.getElementById('toast-styles')) {
        const styles = document.createElement('style');
        styles.id = 'toast-styles';
        styles.textContent = `
            .toast {
                position: fixed;
                bottom: 1.5rem;
                right: 1.5rem;
                padding: 1rem 1.5rem;
                background: var(--bg-primary);
                border-radius: var(--radius-md);
                box-shadow: var(--shadow-lg);
                display: flex;
                align-items: center;
                gap: 0.75rem;
                z-index: 1001;
                animation: slideIn 0.3s ease, slideOut 0.3s ease 2.7s forwards;
                border-left: 4px solid currentColor;
            }
            .toast-info { color: var(--primary); }
            .toast-success { color: #10b981; }
            .toast-warning { color: #f59e0b; }
            .toast-error { color: #ef4444; }
            .toast-message { color: var(--text-primary); }
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(styles);
    }

    document.body.appendChild(toast);

    // Remove after animation
    setTimeout(() => toast.remove(), 3000);
}

function getToastIcon(type) {
    const icons = {
        info: 'ℹ️',
        success: '✅',
        warning: '⚠️',
        error: '❌',
    };
    return icons[type] || icons.info;
}

// ============================================================================
// Initialization
// ============================================================================

function init() {
    initTheme();
    initUpload();
    initQuestion();

    // Theme toggle listener
    elements.themeToggle.addEventListener('click', toggleTheme);

    // Health check
    checkHealth();

    console.log('Medical VQA initialized');
}

// Run on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
