// Quit Game
function quitGame() {
    window.location.href = '/';
}

function validateSelection() {
    // Check if any radio button is selected
    const selectedFrame = document.querySelector('input[name="selected_frame"]:checked');
    if (!selectedFrame) {
        alert("Please select a frame before submitting.");
        return false; // Prevent form submission
    }
    return true; // Allow form submission
}

// Keep track of the selected frame and highlight it
function updateSelection(input) {
    // Deselect any previously selected frames
    const allFrames = document.querySelectorAll('.frame');
    allFrames.forEach(frame => frame.classList.remove('selected'));

    // Highlight the newly selected frame
    const img = input.nextElementSibling; // Image sibling of the radio input
    img.classList.add('selected');
}