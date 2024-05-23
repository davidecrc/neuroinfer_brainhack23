// Event listener for when the DOM content is loaded
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
            const regex = /<a.*?>(.*?)<\/a>/;
            // Populate the select element with the atlas files as options
            atlasFiles.forEach(file => {
                let filename = regex.exec(file)
                filename = filename && filename[1];
                const option = document.createElement('option');
                option.value = filename;
                option.textContent = filename;
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

function fetch_brain_regions(selectedFile) {    // Fetching brain region data from a JSON file
    fetch('../data/' + selectedFile)
    .then(response => response.json())
    .then(data => {
        console.log(data)
        // Extracting brain regions and populating the select element
        var brainRegions = data["data"];
        selected_atlas = data["atlas"];
        var select = document.getElementById('brainRegion');
        select.innerHTML = "";

        brainRegions.forEach(function(region) {
            var option = document.createElement('option');
            option.value = region.id_lab;
            option.textContent = region.lab;
            select.appendChild(option);
        });
    })
    .catch(error => console.error('Error:', error));
}
