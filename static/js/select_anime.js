function unselectAll() {
    const checkboxes = document.querySelectorAll('.anime-list input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
}