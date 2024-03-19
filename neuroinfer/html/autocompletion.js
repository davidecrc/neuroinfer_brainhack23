function autocompleteWords() {
  const textarea = document.getElementById('words');
  const input = textarea.value.trim().toLowerCase();
  const autocompleteContainer = document.getElementById('autocomplete-suggestions');
  autocompleteContainer.innerHTML = '';

  const words = input.split(',').map(word => word.trim());

  if (words.length > 0) {
    const currentWord = words[words.length - 1];
    if (currentWord.length >= 3) {
      const textareaRect = textarea.getBoundingClientRect();
      autocompleteContainer.style.left = `${textareaRect.left}px`;
      autocompleteContainer.style.top = `${textareaRect.bottom}px`;

      fetch('../data/vocabulary7.txt')
        .then(response => response.text())
        .then(data => {
          const dictionaryWords = data.split('\n');
          const matchingWords = dictionaryWords.filter(word =>
            word.toLowerCase().startsWith(currentWord)
          );

          matchingWords.forEach(match => {
            const suggestion = document.createElement('div');
            suggestion.textContent = match;
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

        document.addEventListener("DOMContentLoaded", function() {
            // Get all labels and help containers
            const labels = document.querySelectorAll('.help-caller');
            const helpContainers = document.querySelectorAll('.help-container');

            // Attach mouseover and mouseout events to each label
            labels.forEach((label, index) => {
                label.addEventListener('mouseover', () => {
                    // Hide all help containers
                    helpContainers.forEach(container => container.style.display = 'none');

                    // Display the corresponding help container
                    helpContainers[index].style.display = 'block';

                      const labelsRect = labels[index].getBoundingClientRect();
                      helpContainers[index].style.left = `${labelsRect.left}px`;
                      helpContainers[index].style.top = `${labelsRect.bottom}px`;
                });

                label.addEventListener('mouseout', () => {
                    // Hide the help container when the mouse moves out
                    helpContainers[index].style.display = 'none';
                });
            });
        });

        document.addEventListener("DOMContentLoaded", function() {
            // Get all labels and help containers
            const labels = document.querySelectorAll('.helpdeep-caller');
            const helpContainers = document.querySelectorAll('.deeper_help');

            // Attach click events to each label
            labels.forEach((label, index) => {
                label.addEventListener('click', () => {
                    // Hide all help containers
                    helpContainers.forEach(container => container.style.display = 'none');

                    // Display the corresponding help container
                    helpContainers[index].style.display = 'block';
                });
            });

            // Attach click events to close buttons
            const closeButtons = document.querySelectorAll('.close-button');
            closeButtons.forEach((button) => {
                button.addEventListener('click', () => {
                    // Hide the corresponding help container when the close button is clicked
                    button.parentNode.style.display = 'none';
                });
            });
        });