var params = {}; // same

document.addEventListener("DOMContentLoaded", function() { // var params = []; // params must be declared outside otherwise even the initialization update_papaya_viewer() will work!!!
params["worldSpace"] = true;
params["showControls"] = true;
params["showControlBar"] = true;
params["showOrientation"] = true;
params["combineParametric"] = true;
params["showImageButtons"] = true;
params["mainView"] = 'axial';
params["kioskMode"] = true;
update_papaya_viewer();
});

window.update_papaya_viewer = function(mask = false) {
    if (mask) {
        params["images"] = ["../templates/mni_icbm152_t2_tal_nlin_asym_09c.nii.gz", "../.tmp/mask.nii.gz"];
    } else {
        params["images"] = ["../templates/mni_icbm152_t2_tal_nlin_asym_09c.nii.gz"];
    }
    var visualizer = document.getElementById("visualizer");
    // Create a new div element for Papaya viewer
    papaya.Container.addViewer("visualizer", params);
}


window.submitForm = function() {
      var brainRegion = document.getElementById("brainRegion").value;
      var radius = document.getElementById("radius").value;
      var x = document.getElementById("x").value;
      var y = document.getElementById("y").value;
      var z = document.getElementById("z").value;
      var words = document.getElementById("words").value;
      var probabilities = document.getElementById("probabilities").value;

      var formData = {
        brainRegion: brainRegion,
        radius: radius,
        x: x,
        y: y,
        z: z,
        words: words,
        probabilities: probabilities
      };

      var xhr = new XMLHttpRequest();
      xhr.open("POST", "http://127.0.0.1:5000/", true);
      xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

      xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
          var response = JSON.parse(xhr.responseText);
          displayPlot(response.image);
        }

      };
      var jsonData = JSON.stringify(formData);
      xhr.send(jsonData);
    }

window.displayPlot = function(imageData) {
  update_papaya_viewer(true);
  var graphicsContainer = document.getElementById("graphicsContainer");
  if (graphicsContainer) {
    // Create an image element
    var img = new Image();

    // Set the source of the image to the base64-encoded data
    img.src = 'data:image/png;base64,' + imageData;

    // Set the max-width and max-height properties to fit the container
    img.style.maxWidth = '100%';
    img.style.maxHeight = '100%';

    // Append the image to the graphics container
    graphicsContainer.innerHTML = '';
    graphicsContainer.appendChild(img);
  } else {
    console.error("Graphics container element not found");
  }
}

document.addEventListener('DOMContentLoaded', function() {
  fetch('../data/atlas_list_labels.json')
  .then(response => response.json())
  .then(data => {
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

