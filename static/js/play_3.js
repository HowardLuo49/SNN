function quitGame() {
    window.location.href = '/';
}

function validateSelection() {
    // Check if a frame has already been selected
    const selectedFrame = document.querySelector('input[name="selected_frame"]:checked');
    if (!selectedFrame) {
        alert("Please select a frame before submitting.");
        return false;
    }
    return true;
}

// Keep track of the selected frame and highlight it
function updateSelection(input) {
    // Deselect previously selected frame
    const allFrames = document.querySelectorAll('.frame');
    allFrames.forEach(frame => frame.classList.remove('selected'));

    // Highlight newly selected frame
    const img = input.nextElementSibling;
    img.classList.add('selected');
}