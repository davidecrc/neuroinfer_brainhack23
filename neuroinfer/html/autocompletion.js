function autocompleteWords() {
  const textarea = document.getElementById('words');
  const input = textarea.value.trim().toLowerCase();
  const autocompleteContainer = document.getElementById('autocomplete-suggestions');
  autocompleteContainer.innerHTML = '';

  const words = input.split(',').map(word => word.trim());

  if (words.length > 0) {
    const currentWord = words[words.length - 1];
    if (currentWord.length >= 3) {
      fetch('../data/vocabulary7.txt') // Adjust the path as needed
        .then(response => response.text())
        .then(data => {
          const dictionaryWords = data.split('\n');
          const matchingWords = dictionaryWords.filter(word =>
            word.toLowerCase().startsWith(currentWord)
          );

          matchingWords.forEach(match => {
            const suggestion = document.createElement('div');
            suggestion.textContent = match;

            // Set a random background color for each suggestion
            suggestion.style.backgroundColor = `hsl(${Math.random() * 360}, 70%, 70%)`;

            suggestion.addEventListener('click', () => {
              words[words.length - 1] = currentWord + match.substring(currentWord.length);
              textarea.value = words.join(', ');
              autocompleteContainer.innerHTML = '';
            });

            autocompleteContainer.appendChild(suggestion);
          });

        });
    }
  }
}