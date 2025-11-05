// FILE UPLOAD - SLOT BASED SYSTEM
// This section manages the 4 dedicated file slots for PDF upload

/**
 * Setup file slots with drag-and-drop and click handlers
 */
function setupFileSlots() {
  document.querySelectorAll('.file-slot').forEach(slotElement => {
    const slotKey = slotElement.dataset.slot;
    const input = slotElement.querySelector('.slot-input');
    const dropzone = slotElement.querySelector('.slot-dropzone');
    const placeholder = slotElement.querySelector('.slot-placeholder');
    const fileInfo = slotElement.querySelector('.slot-file-info');
    const removeBtn = slotElement.querySelector('.btn-remove-slot');

    // Click to select file
    dropzone.addEventListener('click', () => {
      if (!window.AppState.fileSlots[slotKey]) {
        input.click();
      }
    });

    // File input change
    input.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        handleSlotFile(slotKey, file, slotElement);
      }
    });

    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, () => {
        dropzone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, () => {
        dropzone.classList.remove('drag-over');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files[0];
      if (file && file.type === 'application/pdf') {
        handleSlotFile(slotKey, file, slotElement);
      } else {
        showToast('Please drop a PDF file', 'error');
      }
    });

    // Remove button
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      removeSlotFile(slotKey, slotElement);
    });
  });

  // Upload all button
  document.getElementById('upload-all-btn').addEventListener('click', handleUploadAll);

  // Clear all button
  document.getElementById('clear-all-slots-btn').addEventListener('click', clearAllSlots);
}

/**
 * Handle file added to slot
 */
function handleSlotFile(slotKey, file, slotElement) {
  // Validate file
  if (file.type !== 'application/pdf') {
    showToast('Only PDF files are allowed', 'error');
    return;
  }

  if (file.size > window.AppState.config.maxFileSize) {
    showToast(`File too large. Max size: ${formatFileSize(window.AppState.config.maxFileSize)}`, 'error');
    return;
  }

  // Store file in state
  window.AppState.fileSlots[slotKey] = file;

  // Update UI
  const placeholder = slotElement.querySelector('.slot-placeholder');
  const fileInfo = slotElement.querySelector('.slot-file-info');
  const statusBadge = slotElement.querySelector('.slot-status');

  placeholder.style.display = 'none';
  fileInfo.style.display = 'block';

  slotElement.classList.add('has-file');
  fileInfo.querySelector('.file-name-display').textContent = file.name;
  fileInfo.querySelector('.file-size-display').textContent = formatFileSize(file.size);

  statusBadge.textContent = '✅ Ready';
  statusBadge.dataset.status = 'filled';

  updateUploadStatus();
}

/**
 * Remove file from slot
 */
function removeSlotFile(slotKey, slotElement) {
  // Clear from state
  window.AppState.fileSlots[slotKey] = null;

  // Update UI
  const placeholder = slotElement.querySelector('.slot-placeholder');
  const fileInfo = slotElement.querySelector('.slot-file-info');
  const statusBadge = slotElement.querySelector('.slot-status');
  const input = slotElement.querySelector('.slot-input');

  placeholder.style.display = 'block';
  fileInfo.style.display = 'none';

  slotElement.classList.remove('has-file');
  input.value = '';

  statusBadge.textContent = '❌ Required';
  statusBadge.dataset.status = 'empty';

  updateUploadStatus();
}

/**
 * Clear all slots
 */
function clearAllSlots() {
  if (!confirm('Remove all files from upload slots?')) return;

  document.querySelectorAll('.file-slot').forEach(slotElement => {
    const slotKey = slotElement.dataset.slot;
    removeSlotFile(slotKey, slotElement);
  });

  showToast('All slots cleared', 'success');
}

/**
 * Update upload status banner
 */
function updateUploadStatus() {
  const filledCount = Object.values(window.AppState.fileSlots).filter(f => f !== null).length;
  const totalCount = Object.keys(window.AppState.fileSlots).length;

  const statusIcon = document.getElementById('upload-status-icon');
  const statusText = document.getElementById('upload-status-text');
  const uploadBtn = document.getElementById('upload-all-btn');
  const clearBtn = document.getElementById('clear-all-slots-btn');
  const statusBanner = document.querySelector('.upload-status-banner');

  statusText.textContent = `${filledCount} of ${totalCount} files uploaded`;

  if (filledCount === totalCount) {
    statusIcon.textContent = '✅';
    statusBanner.classList.add('complete');
    uploadBtn.disabled = false;
  } else {
    statusIcon.textContent = '⏳';
    statusBanner.classList.remove('complete');
    uploadBtn.disabled = true;
  }

  clearBtn.disabled = filledCount === 0;
}

/**
 * Upload all files to server
 */
async function handleUploadAll() {
  const filledCount = Object.values(window.AppState.fileSlots).filter(f => f !== null).length;

  if (filledCount !== 4) {
    showToast('All 4 files must be selected before uploading', 'error');
    return;
  }

  document.getElementById('upload-all-btn').disabled = true;
  document.getElementById('upload-all-btn').textContent = '⏳ Uploading...';

  try {
    // Upload each file to its respective folder
    for (const [slotKey, file] of Object.entries(window.AppState.fileSlots)) {
      if (file) {
        const result = await window.AppState.api.uploadFiles([file], slotKey);

        if (!result.success) {
          throw new Error(`Failed to upload ${file.name}`);
        }

        // Update slot status
        const slotElement = document.querySelector(`[data-slot="${slotKey}"]`);
        const statusBadge = slotElement.querySelector('.slot-status');
        statusBadge.textContent = '✅ Uploaded';
        statusBadge.dataset.status = 'uploaded';
      }
    }

    showToast('All files uploaded successfully!', 'success');
    document.getElementById('upload-all-btn').textContent = '✅ All Files Uploaded';

    // Show hint to run pipeline
    setTimeout(() => {
      if (confirm('Files uploaded! Do you want to go to the Pipeline section to start processing?')) {
        window.location.hash = 'pipeline-section';
      }
    }, 1000);

  } catch (error) {
    console.error('Upload error:', error);
    showToast('Upload failed: ' + error.message, 'error');
    document.getElementById('upload-all-btn').disabled = false;
    document.getElementById('upload-all-btn').textContent = '✅ Upload All Files';
  }
}

// Utility functions
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => toast.classList.add('show'), 10);
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Export functions
if (!window.App) {
  window.App = {};
}

window.App.setupFileSlots = setupFileSlots;
window.App.handleSlotFile = handleSlotFile;
window.App.removeSlotFile = removeSlotFile;
window.App.clearAllSlots = clearAllSlots;
