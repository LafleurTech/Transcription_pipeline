class AudioTranscriptionApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupFileUpload();
        this.updateSubmitButton();
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('process-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.processAudio();
        });

        // Clear results
        document.getElementById('clear-results-btn')?.addEventListener('click', () => {
            this.clearResults();
        });

        // Process more files
        document.getElementById('process-more-btn')?.addEventListener('click', () => {
            this.resetForm();
        });

        // Mode change updates submit button text
        document.getElementById('mode').addEventListener('change', () => {
            this.updateSubmitButton();
        });
    }

    setupFileUpload() {
        const fileInput = document.getElementById('audio-files');
        const fileUploadLabel = document.querySelector('.file-upload-label');
        const fileList = document.getElementById('file-list');

        // File input change
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files);
        });

        // Drag and drop
        fileUploadLabel.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadLabel.classList.add('drag-over');
        });

        fileUploadLabel.addEventListener('dragleave', (e) => {
            e.preventDefault();
            fileUploadLabel.classList.remove('drag-over');
        });

        fileUploadLabel.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadLabel.classList.remove('drag-over');
            this.handleFileSelection(e.dataTransfer.files);
        });
    }

    handleFileSelection(files) {
        const fileList = document.getElementById('file-list');
        const fileArray = Array.from(files);
        
        if (fileArray.length === 0) {
            fileList.style.display = 'none';
            this.updateSubmitButton();
            return;
        }

        // Display selected files
        fileList.innerHTML = '';
        fileArray.forEach((file, index) => {
            const fileItem = this.createFileItem(file, index);
            fileList.appendChild(fileItem);
        });

        fileList.style.display = 'block';
        this.updateSubmitButton();
    }

    createFileItem(file, index) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        const sizeInMB = (file.size / 1024 / 1024).toFixed(1);
        
        fileItem.innerHTML = `
            <i class="fas fa-file-audio"></i>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${sizeInMB} MB</div>
            </div>
            <button type="button" class="remove-file" onclick="audioApp.removeFile(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        return fileItem;
    }

    removeFile(index) {
        const fileInput = document.getElementById('audio-files');
        const dt = new DataTransfer();
        const files = Array.from(fileInput.files);
        
        // Remove the file at the specified index
        files.splice(index, 1);
        
        // Recreate the file list
        files.forEach(file => dt.items.add(file));
        fileInput.files = dt.files;
        
        // Update the display
        this.handleFileSelection(fileInput.files);
    }

    updateSubmitButton() {
        const submitBtn = document.querySelector('.submit-btn');
        const submitText = document.getElementById('submit-text');
        const fileInput = document.getElementById('audio-files');
        const mode = document.getElementById('mode').value;
        
        const modeTexts = {
            'combined': 'Start Complete Analysis',
            'transcription': 'Start Transcription',
            'diarization': 'Start Speaker Identification'
        };
        
        if (fileInput.files.length === 0) {
            submitBtn.disabled = true;
            submitText.textContent = 'Select files to start processing';
        } else {
            submitBtn.disabled = false;
            submitText.textContent = modeTexts[mode] || 'Process Audio';
        }
    }

    async processAudio() {
        const form = document.getElementById('process-form');
        const formData = new FormData(form);
        
        this.showLoading();
        
        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            this.showResults(result);
            
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.hideLoading();
        }
    }

    showLoading() {
        const loading = document.getElementById('loading');
        const mainForm = document.querySelector('.main-form');
        
        loading.style.display = 'flex';
        mainForm.style.display = 'none';
        
        // Simulate progress updates
        this.updateProgress();
    }

    updateProgress() {
        const statusElement = document.getElementById('loading-status');
        const progressFill = document.getElementById('progress-fill');
        
        const steps = [
            { text: 'Uploading files...', progress: 20 },
            { text: 'Initializing AI models...', progress: 40 },
            { text: 'Processing audio...', progress: 70 },
            { text: 'Generating results...', progress: 90 },
            { text: 'Finalizing...', progress: 100 }
        ];
        
        let currentStep = 0;
        
        const updateStep = () => {
            if (currentStep < steps.length) {
                const step = steps[currentStep];
                statusElement.textContent = step.text;
                progressFill.style.width = step.progress + '%';
                currentStep++;
                setTimeout(updateStep, 1000);
            }
        };
        
        updateStep();
    }

    hideLoading() {
        const loading = document.getElementById('loading');
        loading.style.display = 'none';
    }

    showResults(result) {
        const resultsSection = document.getElementById('results');
        const metricsGrid = document.getElementById('results-metrics');
        const resultsContent = document.getElementById('results-content');
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Create metrics
        metricsGrid.innerHTML = `
            <div class="metric-card">
                <h3><i class="fas fa-file-alt"></i> Files Processed</h3>
                <div class="metric-value">${result.files_processed || 0}</div>
            </div>
            <div class="metric-card">
                <h3><i class="fas fa-clock"></i> Processing Time</h3>
                <div class="metric-value">${(result.processing_time || 0).toFixed(1)}</div>
                <div class="metric-unit">seconds</div>
            </div>
            <div class="metric-card">
                <h3><i class="fas fa-cog"></i> Mode</h3>
                <div class="metric-value">${this.formatMode(result.mode)}</div>
            </div>
        `;
        
        // Show detailed results
        this.displayDetailedResults(result.result, resultsContent);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    formatMode(mode) {
        const modes = {
            'transcription': '🎯 Transcription',
            'diarization': '👥 Diarization',
            'combined': '🔄 Combined'
        };
        return modes[mode] || mode;
    }

    displayDetailedResults(results, container) {
        container.innerHTML = '';
        
        if (!results) {
            container.innerHTML = '<p>No results available.</p>';
            return;
        }
        
        if (typeof results === 'object' && !Array.isArray(results)) {
            // Results is an object with filenames as keys
            Object.entries(results).forEach(([filename, content]) => {
                const resultItem = this.createResultItem(filename, content);
                container.appendChild(resultItem);
            });
        } else {
            // Results is a simple value
            const resultItem = this.createResultItem('Results', results);
            container.appendChild(resultItem);
        }
    }

    createResultItem(title, content) {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        const isExpanded = false;
        
        resultItem.innerHTML = `
            <div class="result-header" onclick="this.parentElement.classList.toggle('expanded')">
                <i class="fas fa-file-alt"></i>
                <span>${title}</span>
                <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
            </div>
            <div class="result-content" style="display: none;">
                ${this.formatContent(content)}
            </div>
        `;
        
        // Add click handler to expand/collapse
        const header = resultItem.querySelector('.result-header');
        const contentDiv = resultItem.querySelector('.result-content');
        const chevron = resultItem.querySelector('.fa-chevron-down');
        
        header.addEventListener('click', () => {
            const isVisible = contentDiv.style.display !== 'none';
            contentDiv.style.display = isVisible ? 'none' : 'block';
            chevron.style.transform = isVisible ? 'rotate(0deg)' : 'rotate(180deg)';
        });
        
        return resultItem;
    }

    formatContent(content) {
        if (typeof content === 'string') {
            return `<pre>${this.escapeHtml(content)}</pre>`;
        } else {
            return `<pre>${this.escapeHtml(JSON.stringify(content, null, 2))}</pre>`;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showError(message) {
        const resultsSection = document.getElementById('results');
        const resultsContent = document.getElementById('results-content');
        
        resultsSection.style.display = 'block';
        resultsContent.innerHTML = `
            <div style="background: #fed7d7; color: #c53030; padding: 20px; border-radius: 8px; border-left: 4px solid #e53e3e;">
                <h3><i class="fas fa-exclamation-triangle"></i> Processing Failed</h3>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;
        
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    clearResults() {
        const resultsSection = document.getElementById('results');
        resultsSection.style.display = 'none';
    }

    resetForm() {
        const form = document.getElementById('process-form');
        const fileList = document.getElementById('file-list');
        const resultsSection = document.getElementById('results');
        const mainForm = document.querySelector('.main-form');
        
        form.reset();
        fileList.style.display = 'none';
        fileList.innerHTML = '';
        resultsSection.style.display = 'none';
        mainForm.style.display = 'block';
        
        this.updateSubmitButton();
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.audioApp = new AudioTranscriptionApp();
});

// Add some CSS for drag and drop effects
const style = document.createElement('style');
style.textContent = `
    .file-upload-label.drag-over {
        border-color: #5a67d8 !important;
        background: linear-gradient(135deg, #e6efff 0%, #d6e8ff 100%) !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3) !important;
    }
    
    .result-item.expanded .fa-chevron-down {
        transform: rotate(180deg);
    }
    
    .result-content {
        transition: all 0.3s ease;
    }
`;
document.head.appendChild(style);

// Display selected files
function displaySelectedFiles(input, container) {
    const existingList = container.querySelector('.file-list');
    if (existingList) {
        existingList.remove();
    }
    
    if (input.files.length > 0) {
        const fileList = document.createElement('div');
        fileList.className = 'file-list';
        
        Array.from(input.files).forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <i class="fas fa-file-audio"></i>
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
            `;
            fileList.appendChild(fileItem);
        });
        
        container.appendChild(fileList);
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Show/hide loading spinner
function showLoading() {
    document.getElementById('loading').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

// Show results section
function showResults(data) {
    const resultsSection = document.getElementById('results');
    const resultsContent = document.getElementById('results-content');
    
    // Clear previous results
    resultsContent.innerHTML = '';
    
    // Create results display
    const resultsContainer = document.createElement('div');
    resultsContainer.className = 'results-container';
    
    // Processing info
    const infoDiv = document.createElement('div');
    infoDiv.className = 'processing-info';
    infoDiv.innerHTML = `
        <h3>Processing Complete</h3>
        <p><strong>Mode:</strong> ${data.mode}</p>
        <p><strong>Files Processed:</strong> ${data.files_processed}</p>
        <p><strong>Processing Time:</strong> ${data.processing_time.toFixed(2)}s</p>
        <p><strong>Input Files:</strong> ${data.input_files.join(', ')}</p>
    `;
    resultsContainer.appendChild(infoDiv);
    
    // Results data
    if (data.result) {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-data';
        resultDiv.innerHTML = `
            <h4>Results:</h4>
            <pre>${JSON.stringify(data.result, null, 2)}</pre>
        `;
        resultsContainer.appendChild(resultDiv);
    }
    
    resultsContent.appendChild(resultsContainer);
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Show error message
function showError(message) {
    const resultsSection = document.getElementById('results');
    const resultsContent = document.getElementById('results-content');
    
    resultsContent.innerHTML = `
        <div class="error-message">
            <h3>Error</h3>
            <p>${message}</p>
        </div>
    `;
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Server error');
        }
        
        const data = await response.json();
        showResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Clear results
function clearResults() {
    const resultsSection = document.getElementById('results');
    resultsSection.style.display = 'none';
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupFileUpload();
    
    // Form submission
    document.getElementById('process-form').addEventListener('submit', handleFormSubmit);
    
    // Clear results button
    document.getElementById('clear-results-btn').addEventListener('click', clearResults);
    
    // Health check
    checkServerHealth();
});

// Check server health
async function checkServerHealth() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            console.log('Server is healthy');
        }
    } catch (error) {
        console.warn('Server health check failed:', error);
    }
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

// Show results


// Show error message
function showError(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message error';
    messageDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(messageDiv, container.firstChild);
    
    setTimeout(() => {
        messageDiv.remove();
    }, 5000);
}

// Show success message
function showSuccess(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message success';
    messageDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(messageDiv, container.firstChild);
    
    setTimeout(() => {
        messageDiv.remove();
    }, 5000);
}

// API calls
async function submitTranscription(formData) {
    try {
        const response = await fetch('/transcribe/files', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Transcription failed');
        }
        
        return await response.json();
    } catch (error) {
        throw error;
    }
}

async function submitDiarization(formData) {
    try {
        const response = await fetch('/diarize/files', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Diarization failed');
        }
        
        return await response.json();
    } catch (error) {
        throw error;
    }
}

async function submitLiveTranscription(formData) {
    try {
        const response = await fetch('/transcribe/live', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Live transcription failed');
        }
        
        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        const liveOutput = document.getElementById('live-output');
        const liveResults = document.getElementById('live-results');
        
        liveResults.style.display = 'block';
        liveOutput.innerHTML = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.type === 'final') {
                            showSuccess(`Live transcription completed! Processed ${data.total_chunks} chunks in ${data.processing_time?.toFixed(2)}s`);
                        } else {
                            addLiveChunk(data);
                        }
                    } catch (e) {
                        // Ignore JSON parse errors for incomplete chunks
                    }
                }
            }
        }
        
    } catch (error) {
        throw error;
    }
}

// Add live transcription chunk
function addLiveChunk(data) {
    const liveOutput = document.getElementById('live-output');
    const chunkDiv = document.createElement('div');
    chunkDiv.className = 'live-chunk';
    chunkDiv.innerHTML = `
        <div class="timestamp">Chunk ${data.chunk} - ${new Date(data.timestamp).toLocaleTimeString()}</div>
        <div class="text">${data.text}</div>
    `;
    liveOutput.appendChild(chunkDiv);
    liveOutput.scrollTop = liveOutput.scrollHeight;
}

// Form submissions
document.addEventListener('DOMContentLoaded', function() {
    setupFileUpload();
    
    // Transcription form
    document.getElementById('transcription-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const files = document.getElementById('transcription-files').files;
        
        if (files.length === 0) {
            showError('Please select at least one audio file');
            return;
        }
        
        showLoading();
        
        try {
            const result = await submitTranscription(formData);
            hideLoading();
            showResults(result, 'transcription');
            showSuccess('Transcription completed successfully!');
        } catch (error) {
            hideLoading();
            showError(error.message);
        }
    });
    
    // Diarization form
    document.getElementById('diarization-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const files = document.getElementById('diarization-files').files;
        
        if (files.length === 0) {
            showError('Please select at least one audio file');
            return;
        }
        
        showLoading();
        
        try {
            const result = await submitDiarization(formData);
            hideLoading();
            showResults(result, 'diarization');
            showSuccess('Diarization completed successfully!');
        } catch (error) {
            hideLoading();
            showError(error.message);
        }
    });
    
    // Live transcription form
    document.getElementById('live-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const file = document.getElementById('live-file').files[0];
        
        if (!file) {
            showError('Please select an audio file');
            return;
        }
        
        showLoading();
        
        try {
            await submitLiveTranscription(formData);
            hideLoading();
        } catch (error) {
            hideLoading();
            showError(error.message);
        }
    });
    
    // Clear results
    document.getElementById('clear-results-btn').addEventListener('click', function() {
        document.getElementById('results').style.display = 'none';
        document.getElementById('live-results').style.display = 'none';
    });
});

// Health check on page load
document.addEventListener('DOMContentLoaded', async function() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            console.log('API is healthy');
        }
    } catch (error) {
        showError('Unable to connect to the transcription API. Please check if the server is running.');
    }
});
