let startTime = Date.now();
let flippedCards = [];
let matchedPairs = 0;
const matchedCards = new Set();

// Update timer
setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    document.getElementById('timer').textContent = elapsed;
}, 1000);

// Extract anime name from image src
function getAnimeNameFromSrc(src) {
    const parts = src.split('/');
    const animeDir = parts.slice(-2, -1)[0];
    return decodeURIComponent(animeDir);
}

// Flip card logic
function flipCard(card) {
    const img = card.querySelector('img');

    // Prevent interaction with already matched cards or more than 2 flips
    if (img.hidden && flippedCards.length < 2 && !matchedCards.has(card)) {
        img.hidden = false; // Show the card
        flippedCards.push(card);

        // Check for match
        if (flippedCards.length === 2) {
            const [first, second] = flippedCards;
            const firstAnime = getAnimeNameFromSrc(first.querySelector('img').src);
            const secondAnime = getAnimeNameFromSrc(second.querySelector('img').src);

            if (firstAnime === secondAnime) {
                // Match found: Add to matchedCards set
                matchedCards.add(first);
                matchedCards.add(second);
                flippedCards = [];
                matchedPairs++;

                // Check if all pairs are matched
                if (matchedPairs === 6) {
                    const elapsedTime = Math.floor((Date.now() - startTime) / 1000);
                    window.location.href = `/play_2/result?time=${elapsedTime}`;
                }
            } else {
                // Not a match: Flip back after a delay
                setTimeout(() => {
                    first.querySelector('img').hidden = true;
                    second.querySelector('img').hidden = true;
                    flippedCards = [];
                }, 1000);
            }
        }
    }
}

function restartGame() {
    window.location.reload();
}

function quitGame() {
    window.location.href = '/';
}