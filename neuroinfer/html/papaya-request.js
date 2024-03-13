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
};

// Function to submit a form with a loading overlay
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
};
