// Function to show the overlay
const port = 5000;

function showOverlay() {
  const overlay = document.getElementById("overlay");
  overlay.classList.remove("hidden");
  overlay.classList.add("visible");
}

// Function to hide the overlay
function hideOverlay() {
  const overlay = document.getElementById("overlay");
  overlay.classList.remove("visible");
  overlay.classList.add("hidden");
}

window.downloadMaskFile = function () {
  // Construct the URL to download the file
  const fileUrl = "/.tmp/mask.nii.gz";

  // Create an anchor element
  const anchor = document.createElement("a");
  anchor.style.display = "none";
  document.body.appendChild(anchor);

  // Set the HREF attribute of the anchor element to the file URL
  anchor.href = fileUrl;

  // Set the download attribute to specify the file name
  anchor.download = "mask.nii.gz";

  // Trigger a click event on the anchor element to initiate download
  anchor.click();

  // Clean up: Remove the anchor from the document
  document.body.removeChild(anchor);
};

function updateSecondLoading(progress) {
  var secondLoadingProgress = document.getElementById(
    "second-loading-progress",
  );
  var progressText = document.getElementById("progress-text");
  secondLoadingProgress.style.width = Math.ceil(progress * 100) + "%";
  progressText.textContent = "Progress: " + Math.ceil(progress * 100) + "%";
}

function fetchPercentageProgress() {
  const url = `/.tmp/processing_progress.txt?cacheBuster=${Math.random()}`;
  fetch(url)
    .then((response) => response.text())
    .then((data) => {
      console.log(init_loader);
      //console.log("Content of /.tmp/percentage_progress:", data);
      var progress = parseFloat(data);
      if (isNaN(progress)) {
        init_loader ? updateSecondLoading(0) : updateSecondLoading(1);
      } else {
        updateSecondLoading(progress);
        init_loader = 0;
      }
    })
    .catch((error) => {
      console.error("Error fetching content:", error);
      init_loader ? updateSecondLoading(0) : updateSecondLoading(1);
    });
}

window.rmmaskselector = function (id2rm) {
  var target = document.getElementById(id2rm.id);
  target.remove();
  changeRegionMask();
};

// Function to change the region mask through a POST request
window.changeRegionMask = function () {
  // Extracting the selected brain region from the HTML element
  ids = [];
  var brainRegionArray = document.querySelectorAll(
    'select[name="brainRegion"]',
  );
  for (var i = 0; i < brainRegionArray.length; i++) {
    ids.push(brainRegionArray[i].value);
  }
  smoothfactor = document.getElementById("smooth").value;
  //var brainRegion = document.getElementById("brainRegion").value;

  // Creating form data with the brain region information
  var formData = {
    atlas: selected_atlas,
    brainRegion: ids,
    smooth: smoothfactor,
    func: "update_mask",
  };
  console.log(ids);

  // Creating an XMLHttpRequest for the POST request to the server
  var xhr = new XMLHttpRequest();
  xhr.open("POST", `http://localhost:${port}/`, true);
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

  // Handling the response to update the Papaya viewer
  xhr.onreadystatechange = function () {
    if (xhr.readyState == 4 && xhr.status == 200) {
      update_papaya_viewer(true);
      var response = JSON.parse(xhr.responseText);
      var dispnvx = document.getElementById("totalnumberofvoxel");
      dispnvx.innerHTML = "total voxel inside the mask: " + response.totvx;
      dispnvx.style.backgroundColor = "white";
    }
  };

  var dispnvx = document.getElementById("totalnumberofvoxel");
  dispnvx.innerHTML = "calculating the mask.. ";
  dispnvx.style.backgroundColor = "white";

  // Converting form data to JSON and sending the POST request
  var jsonData = JSON.stringify(formData);
  xhr.send(jsonData);
};

window.addBrainRegion = function () {
  var regContainer = document.getElementById("brainregionselector");
  var regselectors = document.querySelectorAll('select[name="brainRegion"]');
  var selectcontainer = document.createElement("div");
  const node = document.getElementById("brainRegion");
  var select = node.cloneNode(true);
  var rmselect = document.createElement("button");
  selectcontainer.setAttribute("class", "regionselector");
  selectcontainer.setAttribute(
    "id",
    "regionselector" + (regselectors.length + 1),
  );
  selectcontainer.setAttribute("style", "margin-top:5px");
  select.setAttribute("id", "brainRegion" + (regselectors.length + 1));
  select.setAttribute("name", "brainRegion");
  select.setAttribute("onchange", "changeRegionMask()");
  rmselect.setAttribute(
    "onclick",
    "rmmaskselector(" + selectcontainer.id + ")",
  );
  rmselect.setAttribute("class", "removemask");
  rmselect.textContent = "-";
  selectcontainer.appendChild(select);
  selectcontainer.appendChild(rmselect);
  regContainer.appendChild(selectcontainer);
};

// Function to submit a form with a loading overlay
window.submitForm = function () {
  // Extracting form values from HTML elements
  var brainRegion = [];
  var brainRegionArray = document.querySelectorAll(
    'select[name="brainRegion"]',
  );
  for (i = 0; i < brainRegionArray.length; i++) {
    brainRegion.push(brainRegionArray[i].value);
  }
  var radius = document.getElementById("radius").value;
  var x = document.getElementById("x").value;
  var y = document.getElementById("y").value;
  var z = document.getElementById("z").value;
  var words_str = document.getElementById("words").value;
  words = words_str.split(",");
  var probabilities = document.getElementById("probabilities").value;
  var smoothfactor = document.getElementById("smooth").value;

  // Creating form data with the analysis parameters
  var formData = {
    atlas: selected_atlas,
    brainRegion: brainRegion,
    radius: radius,
    x: x,
    y: y,
    z: z,
    words: words_str,
    probabilities: probabilities,
    func: "do_analysis",
    smooth: smoothfactor,
  };

  showOverlay();

  // Call fetchPercentageProgress initially
  fetchPercentageProgress();
  init_loader = 1;
  var intervalId = setInterval(fetchPercentageProgress, 500);

  // Creating an XMLHttpRequest for the POST request to the server
  var xhr = new XMLHttpRequest();
  xhr.open("POST", `http://localhost:${port}/`, true);
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

  // Handling the response after the POST request
  xhr.onreadystatechange = function () {
    if (xhr.readyState == 4) {
      // Removing the overlay once the response is received
      hideOverlay();
      clearInterval(intervalId);

      if (xhr.status == 200) {
        // Parsing the JSON response and displaying the plot
        var response = JSON.parse(xhr.responseText);
        displayPlot(response.image);
        filenames_sys = response.message; // Store filenames globally
        filenames = filenames_sys.map((i) => "/" + i);
        max_value = parseFloat(response.max_value);
        update_papaya_viewer(filenames);
        createRadioButtons();
        createSliceNavigator(words);
        document.getElementById("image-navigator").style.display = "grid";
      }
    }
  };

  // Converting form data to JSON and sending the POST request
  var jsonData = JSON.stringify(formData);
  xhr.send(jsonData);
};

function createRadioButtons() {
  var container = document.getElementById("radio-container");
  container.innerHTML = ""; // Clear previous content

  var emptyCell = document.createElement("div");
  emptyCell.textContent = "";
  emptyCell.setAttribute("class", "grid-item");
  container.appendChild(emptyCell);

  for (var h = 0; h < words.length; h++) {
    // Adjusted to match the number of columns in the table
    var headerCell = document.createElement("div");
    headerCell.textContent = words[h]; // Adjusted index to match words array
    headerCell.setAttribute("class", "grid-item");
    container.appendChild(headerCell);
  }

  var numRows = 3;
  for (var i = 0; i < numRows; i++) {
    var cell = document.createElement("div");
    cell.textContent = "overaly_" + (i + 1); // Adjusted index to match words array
    cell.setAttribute("class", "grid-item");
    container.appendChild(cell);
    for (var j = 0; j < words.length; j++) {
      // Adjusted to match the number of columns in the table
      var cell = document.createElement("div");
      var checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = "file";
      checkbox.value = "row_" + i + "_file_" + (j - 1); // Adjusted index to match filenames array
      checkbox.id = "row_" + i + "_file_" + (j - 1); // Adjusted index to match filenames array
      cell.appendChild(checkbox);
      cell.setAttribute("class", "grid-item");
      container.appendChild(cell);
    }
  }
  var autoRepeat = "";
  for (var i = 0; i < words.length + 1; i++) {
    autoRepeat += "auto ";
  }

  // Apply styles for centering
  container.style.display = "grid";
  container.style.gridTemplateColumns = autoRepeat.trim();
  var supercontainer = document.getElementById("radio-grid-container");
  supercontainer.style.display = "block";
}

function update_overlays() {
  var numRows = 3; // Assuming you have 3 rows of checkboxes

  var checkboxValues = [];

  for (var i = 0; i < numRows; i++) {
    var rowValues = [];
    for (var j = 0; j < words.length; j++) {
      // Assuming `words` is defined globally or within scope
      var checkboxId = "row_" + i + "_file_" + (j - 1); // Adjusted index to match checkbox IDs
      var checkbox = document.getElementById(checkboxId);
      rowValues.push(checkbox.checked); // Pushing boolean value of checkbox
    }
    checkboxValues.push(rowValues);
  }
  console.log(rowValues);
  console.log(checkboxValues[1]);
  console.log(checkboxValues);

  // Creating form data with the brain region information
  var formData = {
    combination_bool: checkboxValues,
    file_list: filenames_sys,
    func: "update_overlays",
  };

  // Creating an XMLHttpRequest for the POST request to the server
  var xhr = new XMLHttpRequest();
  xhr.open("POST", `http://localhost:${port}/`, true);
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

  // Handling the response to update the Papaya viewer
  xhr.onreadystatechange = function () {
    if (xhr.readyState == 4 && xhr.status == 200) {
      var response = JSON.parse(xhr.responseText);
      console.log("response:");
      console.log(response.message);
      let overlay_sys = response.message; // Store filenames globally
      overlay_sys = overlay_sys.map((i) => "/" + i);
      update_papaya_viewer(overlay_sys);
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
  var selectedValue = document.querySelector(
    'input[name="fileSelector"]:checked',
  ).value;
  // Call update_papaya_viewer with the selected filenames element
  update_papaya_viewer(overlays[selectedValue]);
}

window.createSliceNavigator = function (col_names) {
  var slider = document.getElementById("sliders");
  slider.style.display = "block";
  var graphics = document.getElementById("graphics-container");
  graphics.style.display = "block";
  var max_input = document.getElementById("max");
  max_input.value = max_value;

  var container = document.getElementById("image-navigator");
  container.innerHTML = ""; // Clear previous content

  for (var h = 0; h < col_names.length; h++) {
    // Adjusted to match the number of columns in the table
    var headerCell = document.createElement("div");
    headerCell.textContent = col_names[h]; // Adjusted index to match words array
    headerCell.setAttribute("class", "grid-item");
    container.appendChild(headerCell);
  }

  for (var j = 0; j < col_names.length; j++) {
    var cell = document.createElement("div");
    cell.setAttribute("class", "grid-item");
    var radio = document.createElement("input");
    radio.type = "radio";
    radio.id = "radio" + j;
    radio.name = "fileSelector";
    radio.value = "" + j;
    if (j == 0) {
      radio.checked = true;
    }
    radio.addEventListener("change", handleRadioButtonChange);
    cell.appendChild(radio);
    container.appendChild(cell);
  }

  var autoRepeat = "";
  for (var i = 0; i < col_names.length; i++) {
    autoRepeat += "auto ";
  }
  var button = document.createElement("button");
  button.type = "button";
  button.onclick = reset;
  button.innerHTML = "reset";
  container.appendChild(button);
  container.style.gridTemplateColumns = autoRepeat.trim();
  container.style.display = "grid";
};

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
        if (x ** 2 + y ** 2 + z ** 2 <= radius ** 2) {
          count++;
        }
      }
    }
  }
  return count;
}

window.getvsinsphere = function () {
  var radius = document.getElementById("radius").value;
  var container = document.getElementById("totvxinsphere");
  // var totvx = Math.floor((4/3) * Math.PI * Math.pow(parseInt(radius), 3));
  var totvx = voxelsInsideSphere(radius);
  container.innerHTML = "number of voxel in each sphere: " + totvx;
};

// Function to submit a form with a loading overlay
window.load_prev = function () {
  // Extracting form values from HTML elements
  var file2load = document.getElementById("resultsList").value;

  // Creating form data with the analysis parameters
  var formData = {
    loadfile: file2load,
    func: "load_results",
  };

  // Creating an XMLHttpRequest for the POST request to the server
  var xhr = new XMLHttpRequest();
  xhr.open("POST", `http://localhost:${port}/`, true);
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  showOverlay();

  // Handling the response after the POST request
  xhr.onreadystatechange = function () {
    if (xhr.readyState == 4) {
      // Removing the overlay once the response is received

      if (xhr.status == 200) {
        // Parsing the JSON response and displaying the plot
        hideOverlay();
        var response = JSON.parse(xhr.responseText);
        displayPlot(response.image);
        words = response.words;
        filenames_sys = response.message; // Store filenames globally
        filenames = filenames_sys.map((i) => "/" + i);
        max_value = parseFloat(response.max_value);
        update_papaya_viewer(filenames);
        createRadioButtons();
        createSliceNavigator(words);
        document.getElementById("image-navigator").style.display = "grid";
      }
    }
  };

  // Converting form data to JSON and sending the POST request
  var jsonData = JSON.stringify(formData);
  xhr.send(jsonData);
};
