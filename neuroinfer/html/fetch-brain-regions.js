// Event listener for when the DOM content is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Fetching brain region data from a JSON file
    fetch('../data/atlas_list_labels.json')
    .then(response => response.json())
    .then(data => {
        // Extracting brain regions and populating the select element
        var brainRegions = data["har_ox_cort_maxprob-thr25-2mm"];
        var select = document.getElementById('brainRegion');

        brainRegions.forEach(function(region) {
            var option = document.createElement('option');
            option.value = region.id_lab;
            option.textContent = region.lab;
            select.appendChild(option);
        });
    })
    .catch(error => console.error('Error:', error));
});
