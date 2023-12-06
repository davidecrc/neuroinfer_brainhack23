 function submitForm() {
      var brainRegion = document.getElementById("brainRegion").value;
      var x = document.getElementById("x").value;
      var y = document.getElementById("y").value;
      var z = document.getElementById("z").value;
      var words = document.getElementById("words").value;
      var probabilities = document.getElementById("probabilities").value;

      var formData = {
        brainRegion: brainRegion,
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

function displayPlot(imageData) {
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

