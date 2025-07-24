// Tab functionality
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    
    // Hide all tab content
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].classList.remove("active");
    }
    
    // Remove active class from all tab buttons
    tablinks = document.getElementsByClassName("tab-button");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].classList.remove("active");
    }
    
    // Show the specific tab content and mark button as active
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// File upload handling with drag and drop
function setupFileUpload() {
    const fileUploads = document.querySelectorAll('.file-upload');
    
    fileUploads.forEach(upload => {
        const input = upload.querySelector('input[type="file"]');
        const label = upload.querySelector('.file-upload-label');
        
        // Handle file selection
        input.addEventListener('change', function(e) {
            displaySelectedFiles(this, upload);
        });
        
        // Drag and drop functionality
        label.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#5a67d8';
            this.style.background = '#f0f4ff';
        });
        
        label.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = '#667eea';
            this.style.background = '#f8f9ff';
        });
        
        label.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '#667eea';
            this.style.background = '#f8f9ff';
            
            const files = e.dataTransfer.files;
            input.files = files;
            displaySelectedFiles(input, upload);
        });
    });
}

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

// Show results
function showResults(data, type = 'transcription') {
    const resultsSection = document.getElementById('results');
    const resultsContent = document.getElementById('results-content');
    
    let displayContent = '';
    
    if (type === 'transcription') {
        if (data.result && typeof data.result === 'object') {
            displayContent = JSON.stringify(data.result, null, 2);
        } else {
            displayContent = data.result || 'No transcription result available';
        }
    } else if (type === 'diarization') {
        displayContent = JSON.stringify(data, null, 2);
    }
    
    resultsContent.textContent = displayContent;
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

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
