document.addEventListener('DOMContentLoaded', function() {
    // Fetching the list of files from the '../data/' directory
    fetch('../data/')
        .then(response => response.text()) // Convert the response to text
        .then(data => {
            // Split the text response by newline to get a list of files
            const fileList = data.split('\n');

            // Filter the list to include only files containing 'atlas' in their name
            const atlasFiles = fileList.filter(file => file.includes('atlas'));

            // Get the select element to populate with the atlas files
            const select = document.getElementById('atlasList');

            // Populate the select element with the atlas files as options
            atlasFiles.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file;
                select.appendChild(option);
            });

            // Add an event listener to the select element
            select.addEventListener('change', function() {
                // When a file is selected, call the fetch_brain_regions function with the selected file
                const selectedFile = this.value;
                fetch_brain_regions(selectedFile);
            });
        })
        .catch(error => console.error('Error fetching atlas files:', error));
});