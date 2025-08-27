
const pdfUpload = document.getElementById('pdf-upload');
const fileNameSpan = document.getElementById('file-name');
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

let selectedFile = null;

pdfUpload.addEventListener('change', (event) => {
    if (event.target.files.length > 0) {
        selectedFile = event.target.files[0];
        fileNameSpan.textContent = selectedFile.name;
        uploadPdf(selectedFile);
    } else {
        selectedFile = null;
        fileNameSpan.textContent = 'No file chosen';
    }
});

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

function addMessage(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.textContent = message;
    messageDiv.appendChild(contentDiv);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function uploadPdf(file) {
    addMessage('ai', 'Uploading PDF...');
    const formData = new FormData();
    formData.append('pdf', file);

    try {
        const response = await fetch('http://localhost:5000/upload-pdf', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            addMessage('ai', 'PDF uploaded successfully! You can now ask questions.');
        } else {
            const errorData = await response.json();
            addMessage('ai', `Failed to upload PDF: ${errorData.error}. Please try again.`);
        }
    } catch (error) {
        console.error('Error uploading PDF:', error);
        addMessage('ai', 'An error occurred during PDF upload.');
    }
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (message === '') return;

    addMessage('user', message);
    userInput.value = '';

    // Send message to backend
    try {
        const response = await fetch('http://localhost:5000/ask-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: message }),
        });

        if (response.ok) {
            const data = await response.json();
            addMessage('ai', data.answer);
        } else {
            const errorData = await response.json();
            addMessage('ai', `Failed to get an answer: ${errorData.error}. Please try again.`);
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('ai', 'An error occurred while getting an answer.');
    }
}
