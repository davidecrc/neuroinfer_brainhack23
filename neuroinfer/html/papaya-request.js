// Function to change the region mask through a POST request
window.changeRegionMask = function() {
    // Extracting the selected brain region from the HTML element
    ids = [];
    var brainRegionArray = document.querySelectorAll('select[name="brainRegion"]');
    for (var i = 0; i < brainRegionArray.length; i++) {
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
            var response = JSON.parse(xhr.responseText);
            var dispnvx = document.getElementById("totalnumberofvoxel");
            dispnvx.innerHTML = "total voxel inside the mask: " + response.totvx;
            dispnvx.style.backgroundColor = 'white';
        }
    };

    var dispnvx = document.getElementById("totalnumberofvoxel");
    dispnvx.innerHTML = "calculating the mask.. ";
    dispnvx.style.backgroundColor = 'white';

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
    selectcontainer.setAttribute('style', 'margin-top:5px');
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

// Function to submit a form with a loading overlay
window.submitForm = function () {
    // Extracting form values from HTML elements
    var brainRegion = [];
    var brainRegionArray = document.querySelectorAll('select[name="brainRegion"]');
    for (i = 0; i < brainRegionArray.length; i++) {
        brainRegion.push(brainRegionArray[i].value);
    }
    var radius = document.getElementById("radius").value;
    var x = document.getElementById("x").value;
    var y = document.getElementById("y").value;
    var z = document.getElementById("z").value;
    var words_str = document.getElementById("words").value;
    words = words_str.split(',');
    var probabilities = document.getElementById("probabilities").value;
    var smoothfactor = document.getElementById("smooth").value;

    // Creating form data with the analysis parameters
    var formData = {
        brainRegion: brainRegion,
        radius: radius,
        x: x,
        y: y,
        z: z,
        words: words_str,
        probabilities: probabilities,
        func: "do_analysis",
        smooth: smoothfactor
    };

    // Creating an overlay with loading bars
    var overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.top = "0";
    overlay.style.left = "0";
    overlay.style.width = "100%";
    overlay.style.height = "100%";
    overlay.style.backgroundColor = "rgba(255, 255, 255, 0.7)";
    overlay.style.display = "flex";
    overlay.style.flexDirection = "column";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";

    // Creating the first loading bar inside the overlay
    var firstLoadingBar = document.createElement("div");
    firstLoadingBar.style.border = "4px solid #3498db";
    firstLoadingBar.style.borderRadius = "50%";
    firstLoadingBar.style.borderTop = "4px solid #ffffff";
    firstLoadingBar.style.width = "40px";
    firstLoadingBar.style.height = "40px";
    firstLoadingBar.style.animation = "spin 1s linear infinite";
    overlay.appendChild(firstLoadingBar);

    // Creating the second loading bar inside the overlay
    var secondLoadingBar = document.createElement("div");
    secondLoadingBar.style.marginTop = "20px";
    secondLoadingBar.style.width = "100%";
    secondLoadingBar.style.backgroundColor = "#f3f3f3";
    secondLoadingBar.style.height = "20px";
    overlay.appendChild(secondLoadingBar);

    // Creating the progress bar inside the second loading bar
    var secondLoadingProgress = document.createElement("div");
    secondLoadingProgress.id = "second-loading-progress";
    secondLoadingProgress.style.width = "0%";
    secondLoadingProgress.style.backgroundColor = "#3498db";
    secondLoadingProgress.style.height = "100%";
    secondLoadingBar.appendChild(secondLoadingProgress);

    // Creating the progress text below the second loading bar
    var progressText = document.createElement("div");
    progressText.style.marginTop = "20px";
    progressText.id = "progress-text";
    progressText.style.marginTop = "5px";
    progressText.style.fontSize = "14px";
    progressText.style.textAlign = "center";
    overlay.appendChild(progressText);

    // Appending the overlay to the body
    document.body.appendChild(overlay);

    function updateSecondLoading(progress) {
        var secondLoadingProgress = document.getElementById('second-loading-progress');
        var progressText = document.getElementById('progress-text');
        secondLoadingProgress.style.width = Math.ceil(progress*100) + '%';
        progressText.textContent = 'Progress: ' + Math.ceil(progress*100) + '%';
    }

    function fetchPercentageProgress() {
        const url = `/.tmp/processing_progress.txt?cacheBuster=${Math.random()}`;
        fetch(url)
            .then(response => response.text())
            .then(data => {
                console.log(init_loader);
                //console.log("Content of /.tmp/percentage_progress:", data);
                var progress = parseFloat(data);
                if (isNaN(progress)) {
                    (init_loader) ? updateSecondLoading(0) : updateSecondLoading(1);
                }
                else {
                    updateSecondLoading(progress);
                    init_loader = 0;
                }

            })
            .catch(error => {
                console.error('Error fetching content:', error);
                (init_loader) ? updateSecondLoading(0) : updateSecondLoading(1);
            });
    }

    // Call fetchPercentageProgress initially
    fetchPercentageProgress();
    init_loader=1;
    var intervalId = setInterval(fetchPercentageProgress, 500);

    // Creating an XMLHttpRequest for the POST request to the server
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://127.0.0.1:5000/", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    // Handling the response after the POST request
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4) {
            // Removing the overlay once the response is received
            document.body.removeChild(overlay);
            clearInterval(intervalId);

            if (xhr.status == 200) {
                // Parsing the JSON response and displaying the plot
                var response = JSON.parse(xhr.responseText);
                displayPlot(response.image);
                filenames = response.message; // Store filenames globally
                filenames = filenames.map(i => '/' + i);
                update_papaya_viewer(filenames);
                createRadioButtons();
                createSliceNavigator(words);
                document.getElementById("image-navigator").style.display = 'grid';
            }
        }
    };

    // Converting form data to JSON and sending the POST request
    var jsonData = JSON.stringify(formData);
    xhr.send(jsonData);
};


function createRadioButtons() {
    var container = document.getElementById('radio-container');
    container.innerHTML = ''; // Clear previous content

    var emptyCell = document.createElement('div');
    emptyCell.textContent = '';
    emptyCell.setAttribute('class', 'grid-item')
    container.appendChild(emptyCell);

    for (var h = 0; h < words.length; h++) { // Adjusted to match the number of columns in the table
        var headerCell = document.createElement('div');
        headerCell.textContent = words[h]; // Adjusted index to match words array
        headerCell.setAttribute('class', 'grid-item')
        container.appendChild(headerCell);
    }

    var numRows = 3;
    for (var i = 0; i < numRows; i++) {
        var cell = document.createElement('div');
        cell.textContent = 'overaly_' + (i + 1); // Adjusted index to match words array
        cell.setAttribute('class', 'grid-item')
        container.appendChild(cell);
        for (var j = 0; j < words.length; j++) { // Adjusted to match the number of columns in the table
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
    for (var i = 0; i < words.length + 1; i++) {
        autoRepeat += "auto ";
    }

    // Apply styles for centering
    container.style.display = 'grid';
    container.style.gridTemplateColumns = autoRepeat.trim();
    var supercontainer = document.getElementById('radio-grid-container');
    supercontainer.style.display = 'block';

}


function update_overlays() {
    var numRows = 3; // Assuming you have 3 rows of checkboxes

    var checkboxValues = [];

    for (var i = 0; i < numRows; i++) {
        var rowValues = [];
        for (var j = 0; j < words.length; j++) { // Assuming `words` is defined globally or within scope
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
            createSliceNavigator(["Overlay 1", "Overlay 2", "Overlay 3"]);
        }
    };

    // Converting form data to JSON and sending the POST request
    var jsonData = JSON.stringify(formData);
    xhr.send(jsonData);
}

window.reset = function reset() {
update_papaya_viewer(filenames);
createSliceNavigator(words);
};

function handleRadioButtonChange() {
    var selectedValue = document.querySelector('input[name="fileSelector"]:checked').value;
    // Call update_papaya_viewer with the selected filenames element
    update_papaya_viewer(overlays[selectedValue]);
}

window.createSliceNavigator = function (col_names) {
    var container = document.getElementById('image-navigator');
    container.innerHTML = ''; // Clear previous content

    for (var h = 0; h < col_names.length; h++) { // Adjusted to match the number of columns in the table
        var headerCell = document.createElement('div');
        headerCell.textContent = col_names[h]; // Adjusted index to match words array
        headerCell.setAttribute('class', 'grid-item')
        container.appendChild(headerCell);
    }

    for (var j = 0; j < col_names.length; j++) {
        var cell = document.createElement('div');
        cell.setAttribute('class', 'grid-item');
        var radio = document.createElement('input');
        radio.type = 'radio';
        radio.id = "radio" + j;
        radio.name = "fileSelector";
        radio.value = "" + j;
        if (j == 0) {
            radio.checked = true;
        }
        radio.addEventListener('change', handleRadioButtonChange);
        cell.appendChild(radio)
        container.appendChild(cell)
    }

    var autoRepeat = "";
    for (var i = 0; i < col_names.length; i++) {
        autoRepeat += "auto ";
    }
    var button = document.createElement('button');
    button.type = "button";
    button.onclick = reset;
    button.innerHTML = "reset";
    container.appendChild(button);
    container.style.gridTemplateColumns = autoRepeat.trim();
    container.style.display = 'grid';
}

function voxelsInsideSphere(radius) {
    // Calculate the dimensions of the bounding box
    let sideLength = Math.ceil(2 * radius); // Round up to ensure the sphere is fully contained
    // Initialize counter for voxels inside the sphere
    let count = 0;
    // Iterate over all voxels in the bounding box
    for (let x = -sideLength; x <= sideLength; x++) {
        for (let y = -sideLength; y <= sideLength; y++) {
            for (let z = -sideLength; z <= sideLength; z++) {
                // Check if the voxel is inside the sphere using the sphere equation
                if (x**2 + y**2 + z**2 <= radius**2) {
                    count++;
                }
            }
        }
    }
    return count;
}

window.getvsinsphere = function() {
    var radius = document.getElementById("radius").value;
    var container = document.getElementById('totvxinsphere');
    // var totvx = Math.floor((4/3) * Math.PI * Math.pow(parseInt(radius), 3));
    var totvx = voxelsInsideSphere(radius);
    container.innerHTML = "number of voxel in each sphere: " + totvx;
}