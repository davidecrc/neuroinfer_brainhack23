// Function to change the region mask through a POST request
window.changeRegionMask = function() {
    // Extracting the selected brain region from the HTML element
    ids = [];
    var brainRegionArray = document.querySelectorAll('select[name="brainRegion"]');
    for (i = 0; i < brainRegionArray.length; i++) {
        ids.push(brainRegionArray[i].value);
    }
    smoothfactor = document.getElementById("smooth").value;
    //var brainRegion = document.getElementById("brainRegion").value;

    // Creating form data with the brain region information
    var formData = {
        brainRegion: ids,
        smooth: smoothfactor,
        func: "update_mask"
    };
    console.log(ids)

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

window.addBrainRegion = function() {
    var regContainer = document.getElementById('brainregionselector');
    var regselectors = document.querySelectorAll('select[name="brainRegion"]');
    var selectcontainer = document.createElement('div');
    const node = document.getElementById('brainRegion');
    var select = node.cloneNode(true);
    var rmselect = document.createElement('button');
    selectcontainer.setAttribute('class', 'regionselector');
    selectcontainer.setAttribute('id', 'regionselector' + (regselectors.length + 1));
    select.setAttribute('id', "brainRegion" + (regselectors.length + 1));
    select.setAttribute('name', "brainRegion");
    select.setAttribute('onchange', "changeRegionMask()");
    rmselect.setAttribute('onclick', "rmmaskselector(" + selectcontainer.id + ")");
    rmselect.setAttribute('class', "removemask");
    rmselect.textContent = "-";
    selectcontainer.appendChild(select);
    selectcontainer.appendChild(rmselect);
    regContainer.appendChild(selectcontainer);
}

window.rmmaskselector = function(id2rm) {
    var target = document.getElementById(id2rm.id);
    target.remove();
    changeRegionMask();
}

// Function to submit a form with a loading overlay
window.submitForm = function () {
    // Extracting form values from HTML elements
    var brainRegion = document.getElementById("brainRegion").value;
    var radius = document.getElementById("radius").value;
    var x = document.getElementById("x").value;
    var y = document.getElementById("y").value;
    var z = document.getElementById("z").value;
    words = document.getElementById("words").value;
    var probabilities = document.getElementById("probabilities").value;
    smoothfactor = document.getElementById("smooth").value;

    // Creating form data with the analysis parameters
    var formData = {
        brainRegion: brainRegion,
        radius: radius,
        x: x,
        y: y,
        z: z,
        words: words,
        probabilities: probabilities,
        func: "do_analysis",
        smooth: smoothfactor
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
                filenames = response.message; // Store filenames globally
                update_papaya_viewer(response.message);
                createRadioButtons();
                document.getElementById("image-navigator").style.display = 'grid';
            }
        }
    };

    // Converting form data to JSON and sending the POST request
    var jsonData = JSON.stringify(formData);
    xhr.send(jsonData);
};

function createRadioButtons() {
    var words_split = words.split(",");
    var container = document.getElementById('radio-container');
    container.innerHTML = ''; // Clear previous content

    var emptyCell = document.createElement('div');
    emptyCell.textContent = '';
    emptyCell.setAttribute('class', 'grid-item')
    container.appendChild(emptyCell);

    for (var h = 0; h < words_split.length; h++) { // Adjusted to match the number of columns in the table
        var headerCell = document.createElement('div');
        headerCell.textContent = words_split[h]; // Adjusted index to match words array
        headerCell.setAttribute('class', 'grid-item')
        container.appendChild(headerCell);
    }

    var numRows = 3;
    for (var i = 0; i < numRows; i++) {
        var cell = document.createElement('div');
        cell.textContent = 'overaly_' + (i + 1); // Adjusted index to match words array
        cell.setAttribute('class', 'grid-item')
        container.appendChild(cell);
        for (var j = 0; j < words_split.length; j++) { // Adjusted to match the number of columns in the table
            var cell = document.createElement('div');
                var checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.name = 'file';
                checkbox.value = 'row_' + i + '_file_' + (j - 1); // Adjusted index to match filenames array
                checkbox.id = 'row_' + i + '_file_' + (j - 1); // Adjusted index to match filenames array
                cell.appendChild(checkbox);
                cell.setAttribute('class', 'grid-item')
            container.appendChild(cell);
        }
    }
    var autoRepeat = "";
    for (var i = 0; i < words_split.length + 1; i++) {
        autoRepeat += "auto ";
    }

    // Apply styles for centering
    container.style.display = 'grid';
    container.style.gridTemplateColumns = autoRepeat.trim();
    var supercontainer = document.getElementById('radio-grid-container');
    supercontainer.style.display = 'block';

}


function update_overlays() {
    var words_split = words.split(",");
    var container = document.getElementById('radio-container');
    var numRows = 3; // Assuming you have 3 rows of checkboxes

    var checkboxValues = [];

    for (var i = 0; i < numRows; i++) {
        var rowValues = [];
        for (var j = 0; j < words_split.length; j++) { // Assuming `words` is defined globally or within scope
            var checkboxId = 'row_' + i + '_file_' + (j - 1); // Adjusted index to match checkbox IDs
            var checkbox = document.getElementById(checkboxId);
            rowValues.push(checkbox.checked); // Pushing boolean value of checkbox
        }
        checkboxValues.push(rowValues);
    }
    console.log(rowValues)
    console.log(checkboxValues[1])
    console.log(checkboxValues)

    // Creating form data with the brain region information
    var formData = {
        combination_bool: checkboxValues,
        file_list: filenames,
        func: "update_overlays"
    };

    // Creating an XMLHttpRequest for the POST request to the server
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://127.0.0.1:5000/", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    // Handling the response to update the Papaya viewer
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            var response = JSON.parse(xhr.responseText);
            console.log("response:")
            console.log(response.message)
            update_papaya_viewer(response.message);
        }
    };

    // Converting form data to JSON and sending the POST request
    var jsonData = JSON.stringify(formData);
    xhr.send(jsonData);
}

window.reset = function reset() {
update_papaya_viewer(filenames);
};
