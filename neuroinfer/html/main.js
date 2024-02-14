// Variable to store parameters for Papaya viewer
var params = {};

// Event listener for when the DOM content is loaded
document.addEventListener("DOMContentLoaded", function() {
    // Setting default parameters for the Papaya viewer
    params["worldSpace"] = true;
    params["showControls"] = true;
    params["showControlBar"] = true;
    params["showOrientation"] = true;
    params["combineParametric"] = true;
    params["showImageButtons"] = true;
    params["mainView"] = 'axial';
    params["kioskMode"] = true;

    // Initializing the Papaya viewer with the specified parameters
    update_papaya_viewer();
});

// Function to update the Papaya viewer with or without a mask
window.update_papaya_viewer = function(mask = false) {
    // Setting the images for the Papaya viewer based on the presence of a mask
    if (mask) {
        params["images"] = ["../templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz", "../.tmp/mask.nii.gz"];
    } else {
        params["images"] = ["../templates/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"];
    }

    // Getting the visualizer element and creating a new Papaya viewer
    var visualizer = document.getElementById("visualizer");
    papaya.Container.addViewer("visualizer", params);
}

// Function to change the region mask through a POST request
window.changeRegionMask = function() {
    // Extracting the selected brain region from the HTML element
    var brainRegion = document.getElementById("brainRegion").value;

    // Creating form data with the brain region information
    var formData = {
        brainRegion: brainRegion
    };

    // Creating an XMLHttpRequest for the POST request to the server
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://127.0.0.1:5000/", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    // Handling the response to update the Papaya viewer
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            update_papaya_viewer(true);
        }
    };

    // Converting form data to JSON and sending the POST request
    var jsonData = JSON.stringify(formData);
    xhr.send(jsonData);
}

// Function to submit a form with a loading overlay
window.submitForm = function() {
    // Creating an overlay with a loading bar
    var overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.top = "0";
    overlay.style.left = "0";
    overlay.style.width = "100%";
    overlay.style.height = "100%";
    overlay.style.backgroundColor = "rgba(255, 255, 255, 0.7)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";

    // Creating a loading bar inside the overlay
    var loadingBar = document.createElement("div");
    loadingBar.style.border = "4px solid #3498db";
    loadingBar.style.borderRadius = "50%";
    loadingBar.style.borderTop = "4px solid #ffffff";
    loadingBar.style.width = "40px";
    loadingBar.style.height = "40px";
    loadingBar.style.animation = "spin 1s linear infinite";

    // Appending the loading bar to the overlay
    overlay.appendChild(loadingBar);
    document.body.appendChild(overlay);

    // Simulating a 3-second delay before sending a POST request
    setTimeout(function() {
        // Creating an XMLHttpRequest for the POST request to the server
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "http://127.0.0.1:5000/", true);
        xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

        // Handling the response after the POST request
        xhr.onreadystatechange = function () {
            if (xhr.readyState == 4) {
                // Removing the overlay once the response is received
                document.body.removeChild(overlay);

                if (xhr.status == 200) {
                    // Parsing the JSON response and displaying the plot
                    var response = JSON.parse(xhr.responseText);
                    displayPlot(response.image);
                }
            }
        };

        // Converting form data to JSON and sending the POST request
        var jsonData = JSON.stringify(formData);
        xhr.send(jsonData);
    }, 3000); // 3000 milliseconds (3 seconds)
};

// CSS animation for the loading bar
var style = document.createElement("style");
style.type = "text/css";
style.innerHTML = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);

// Function to submit form data for analysis
window.submitForm = function() {
    // Extracting form values from HTML elements
    var brainRegion = document.getElementById("brainRegion").value;
    var radius = document.getElementById("radius").value;
    var x = document.getElementById("x").value;
    var y = document.getElementById("y").value;
    var z = document.getElementById("z").value;
    var words = document.getElementById("words").value;
    var probabilities = document.getElementById("probabilities").value;

    // Creating form data with the analysis parameters
    var formData = {
        brainRegion: brainRegion,
        radius: radius,
        x: x,
        y: y,
        z: z,
        words: words,
        probabilities: probabilities
    };

    // Creating an XMLHttpRequest for the POST request to the server
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://127.0.0.1:5000/", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    // Handling the response after the POST request
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // Parsing the JSON response and displaying the plot
            var response = JSON.parse(xhr.responseText);
            displayPlot(response.image);
        }
    };

    // Converting form data to JSON and sending the POST request
    var jsonData = JSON.stringify(formData);
    xhr.send(jsonData);
}

// Function to display a plot in the Papaya viewer
window.displayPlot = function(imageData) {
    // Updating the Papaya viewer with the new data
    update_papaya_viewer(true);

    // Getting the graphics container element
    var graphicsContainer = document.getElementById("graphicsContainer");

    // Checking if the graphics container element exists
    if (graphicsContainer) {
        // Creating an image element
        var img = new Image();

        // Setting the source of the image to the base64-encoded data
        img.src = 'data:image/png;base64,' + imageData;

        // Setting max-width and max-height properties to fit the container
        img.style.maxWidth = '100%';
        img.style.maxHeight = '100%';

        // Appending the image to the graphics container
        graphicsContainer.innerHTML = '';
        graphicsContainer.appendChild(img);
    } else {
        // Logging an error if the graphics container element is not found
        console.error("Graphics container element not found");
    }
}

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
